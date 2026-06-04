import torch
import numpy as np
import os
import functools
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
# --- IMPORTS FOR MOLECULE BUILDING ---
from src.models.diffsbdd.analysis.molecule_builder import build_molecule, process_molecule


def dict_to_namespace(d: dict) -> Namespace:
    """Recursively convert a nested dict (e.g. from yaml.safe_load) to a Namespace.

    This lets you write config.ppo.lr instead of config["ppo"]["lr"] everywhere
    in the codebase. A single shared function avoids the RecursiveNamespace class
    that the old generation scripts each defined for themselves.
    """
    ns = Namespace()
    for key, value in d.items():
        setattr(ns, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return ns

@functools.lru_cache(maxsize=4096)
def find_gt_files(sample_name: str, root_dir: str) -> Tuple[Path, Path]:
    """
    Return (ligand.sdf, pocket.pdb) paths for *sample_name* relative to *root_dir*
    which comes from config.yaml file.
    VERBOSE DEBUGGING ENABLED.
    """
    root = Path(root_dir)
    
    # 1. Extract base name
    if '_pocket.pdb' in sample_name:
        base_name = sample_name.split('_pocket.pdb')[0]
    elif '_pocket_only.pdb' in sample_name: 
        base_name = sample_name.split('_pocket_only.pdb')[0]
    else:
        base_name = sample_name
        
    base_name = Path(base_name).name

    # 2. Construct Paths
    root_parent = root.parent
    # Try preprocessed folder structure first
    ligand_path = root_parent / "02_preprocessed" / "sdf_files" / f"{base_name}.sdf"
    pocket_path = root_parent / "02_preprocessed" / "pocket_files" / f"{base_name}_pocket.pdb"

    return ligand_path, pocket_path


def get_reference_ligand(name: str, dataset_info: dict) -> Optional[Chem.Mol]:
    """
    Loads and centers the reference ligand using rigid translation.
    """
    if dataset_info is None or 'datadir' not in dataset_info:
        print(f"[ERR] get_reference_ligand: 'datadir' missing from dataset_info")
        return None

    data_root = dataset_info['datadir']
    
    ligand_path, pocket_path = find_gt_files(name, str(data_root))
    
    if ligand_path is None or pocket_path is None:
        return None

    try:
        _, ref_mol = center_pocket_on_ligand_com(str(pocket_path), str(ligand_path))
        return ref_mol
    except Exception as e:
        print(f"[ERR] Failed to center/load reference for {name}: {e}")
        return None


def center_pocket_on_ligand_com(pocket_pdb_path: str, ref_ligand_sdf_path: str):
    """
    Centers a protein pocket and a reference ligand based on the ligand's 
    center of mass (CoM). Rigid translation only.
    """
    # 1. Load the reference ligand
    ref_mol_supplier = Chem.SDMolSupplier(ref_ligand_sdf_path, removeHs=False)
    ref_mol = next((m for m in ref_mol_supplier if m is not None), None)
    if ref_mol is None:
        raise ValueError(f"Could not load a valid molecule from {ref_ligand_sdf_path}")

    # 2. Load the pocket
    pocket_parser = PDBParser(QUIET=True)
    try:
        pocket_structure = pocket_parser.get_structure("pocket", pocket_pdb_path)
    except Exception:
        return None, None

    # 3. Calculate CoM
    lig_coords = ref_mol.GetConformer(0).GetPositions()
    lig_com = np.mean(lig_coords, axis=0)
    

    # 4. Center the pocket
    for atom in pocket_structure.get_atoms():
        atom.set_coord(atom.get_coord() - lig_com)

    # 5. Center the reference ligand (Rigid Translation)
    centered_ref_conf = Chem.Conformer(ref_mol.GetNumAtoms())
    for i in range(ref_mol.GetNumAtoms()):
        new_pos = lig_coords[i] - lig_com
        centered_ref_conf.SetAtomPosition(i, new_pos)
    
    ref_mol.RemoveAllConformers()
    ref_mol.AddConformer(centered_ref_conf)

    # --- DEBUG: VERIFY CENTERING ---
    new_coords = ref_mol.GetConformer(0).GetPositions()
    new_com = np.mean(new_coords, axis=0)

    return pocket_structure, ref_mol


def center_ligand_on_com(ligand_sdf_path: str):
    """Just center the ligand, don't need pocket."""
    mol = SDMolSupplier(ligand_sdf_path, removeHs=False)[0]
    coords = mol.GetConformer(0).GetPositions()
    com = np.mean(coords, axis=0)
    
    # Center conformer
    centered_conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        new_pos = coords[i] - com
        centered_conf.SetAtomPosition(i, new_pos)
    
    mol.RemoveAllConformers()
    mol.AddConformer(centered_conf)
    return mol


def batch_to_list(data, batch_mask):
    """Helper to split batched tensors."""
    unique_ids = np.unique(batch_mask.cpu()) if hasattr(batch_mask, 'cpu') else np.unique(batch_mask)
    return [data[batch_mask == bid] for bid in unique_ids]


def build_molecules_from_batch(xh_lig: torch.Tensor,
                               lig_mask: torch.Tensor,
                               dataset_info,
                               ddpm_module=None,
                               reconstruction_fn=None) -> Tuple[List, Dict]:
    """
    Build RDKit molecule objects from batched ligand tensors.
    """
    # Handle Virtual Nodes
    if ddpm_module and getattr(ddpm_module, 'virtual_nodes', False):
        atom_types = xh_lig[:, 3:].argmax(1)
        vnode_mask = (atom_types == ddpm_module.virtual_atom)
        xh_lig = xh_lig[~vnode_mask]
        lig_mask = lig_mask[~vnode_mask]
        
        if xh_lig.shape[0] == 0:
            return [], {}

    # Move to CPU
    x = xh_lig[:, :3].detach().cpu()
    atom_type = torch.argmax(xh_lig[:, 3:], dim=1).detach().cpu()
    lig_mask = lig_mask.cpu()

    molecules = []
    molecule_to_batch_idx = {}

    # Build
    coords_list = batch_to_list(x, lig_mask)
    atoms_list = batch_to_list(atom_type, lig_mask)

    for batch_idx, (coords, atoms) in enumerate(zip(coords_list, atoms_list)):
        try:
            if reconstruction_fn is not None:
                mol = reconstruction_fn(coords, atoms)
            else:
                mol = build_molecule(coords, atoms, dataset_info, add_coords=True)
                mol = process_molecule(
                    mol,
                    add_hydrogens=False,
                    sanitize=True,
                    relax_iter=0,
                    largest_frag=True
                )
            if mol is not None:
                molecules.append(mol)
                molecule_to_batch_idx[len(molecules) - 1] = batch_idx
        except Exception as e:
            continue

    return molecules, molecule_to_batch_idx

def permute_timesteps(rollout_data, device):
    """
    Randomly permute diffusion timesteps *per molecule* for:
        - molecule-wise tensors  [B, T]
        - atom-wise  tensors     [N, T, F]   (use lig_mask for lookup)

    Adds no CPU sync, no Python loops, O(N+T) memory.
    """
    B, T = rollout_data["timesteps"].shape
    perms = torch.stack([torch.randperm(T, device=device) for _ in range(B)])  # [B, T]

    # ---------- molecule-wise tensors ----------
    for key in ("timesteps", "old_log_probs"):
        if key in rollout_data and rollout_data[key] is not None:
            rollout_data[key] = rollout_data[key].gather(1, perms)

    # ---------- atom-wise tensors --------------
    for key in ("latents", "next_latents"):
        if key not in rollout_data or rollout_data[key] is None:
            continue

        x        = rollout_data[key]                     # [N, T, F]
        lig_mask = rollout_data["masks"][0]              # [N]  global IDs
        N, _, F  = x.shape

        # build lookup: global-ID → local batch idx (0..B-1)
        ids          = torch.unique(lig_mask)
        id2local     = {int(gid): idx for idx, gid in enumerate(ids.tolist())}
        local_idx    = torch.tensor([id2local[int(g)] for g in lig_mask.tolist()],
                                    device=device, dtype=torch.long)          # [N]

        # perms[local_idx] gives a [N, T] index tensor
        gather_idx = perms[local_idx]          # [N, T]

        # expand for feature dim and gather
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, F)   # [N, T, F]
        rollout_data[key] = x.gather(1, gather_idx)

    return rollout_data

def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    # with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m)