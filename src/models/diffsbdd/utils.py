from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import networkx as nx
from networkx.algorithms import isomorphism
from Bio.PDB.Polypeptide import is_aa
from pathlib import Path
import functools
import glob
from typing import Tuple, Iterable, Any, Union, Optional
from Bio.PDB import PDBParser, Atom, PDBIO, Model
import os
from io import StringIO
import re
from Bio.PDB.Atom import DisorderedAtom




class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


#####


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        norm_type: float = 2.0) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in
         parameters]), norm_type)

    return total_norm


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    #with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m)

    # print(f'Wrote SDF file to {sdf_path}')


def residues_to_atoms(x_ca, atom_encoder):
    x = x_ca
    one_hot = F.one_hot(
        torch.tensor(atom_encoder['C'], device=x_ca.device),
        num_classes=len(atom_encoder)
    ).repeat(*x_ca.shape[:-1], 1)
    return x, one_hot


def get_residue_with_resi(pdb_chain, resi):
    res = [x for x in pdb_chain.get_residues() if x.id[1] == resi]
    assert len(res) == 1
    return res[0]


def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):

    if ligand.endswith(".sdf"):
        # ligand as sdf file
        rdmol = Chem.SDMolSupplier(str(ligand))[0]
        ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        resi = None
    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ligand.split(':')
        ligand = get_residue_with_resi(pdb_model[chain], int(resi))
        ligand_coords = torch.from_numpy(
            np.array([a.get_coord() for a in ligand.get_atoms()]))

    pocket_residues = []
    for residue in pdb_model.get_residues():
        if residue.id[1] == resi:
            continue  # skip ligand itself

        res_coords = torch.from_numpy(
            np.array([a.get_coord() for a in residue.get_atoms()]))
        if is_aa(residue.get_resname(), standard=True) \
                and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
            pocket_residues.append(residue)

    return pocket_residues


def batch_to_list(data, batch_mask):
    # data_list = []
    # for i in torch.unique(batch_mask):
    #     data_list.append(data[batch_mask == i])
    # return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def num_nodes_to_batch_mask(n_samples, num_nodes, device=None):
    """
    Create a batch mask given the number of nodes per sample.
    
    Args:
        n_samples: Number of samples in the batch
        num_nodes: Either a single integer (same number of nodes for all samples)
                  or a tensor/list with n_samples elements
        device: The device to put the tensors on
    
    Returns:
        Tensor with repeated indices indicating batch membership
    """
    # Handle device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate input
    if isinstance(num_nodes, int):
        # Same number of nodes for all samples
        num_nodes_per_sample = torch.full((n_samples,), num_nodes, 
                                         device=device, dtype=torch.long)
    elif isinstance(num_nodes, torch.Tensor):
        # Convert tensor to long if needed
        if num_nodes.dtype != torch.long and num_nodes.dtype != torch.int:
            num_nodes_per_sample = num_nodes.to(torch.long)
        else:
            num_nodes_per_sample = num_nodes
        
        # Move to the correct device
        num_nodes_per_sample = num_nodes_per_sample.to(device)
    else:
        # Convert list to tensor
        num_nodes_per_sample = torch.tensor(num_nodes, device=device, dtype=torch.long)
    
    # Create batch indices
    sample_inds = torch.arange(n_samples, device=device)
    
    # Create mask by repeating indices
    return torch.repeat_interleave(sample_inds, num_nodes_per_sample)


def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    for atom in rdmol.GetAtoms():
        # Add the atoms as nodes
        graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph


def calc_rmsd(mol_a, mol_b):
    """ Calculate RMSD of two molecules with unknown atom correspondence. """
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = isomorphism.GraphMatcher(
        graph_a, graph_b,
        node_match=lambda na, nb: na['atom_type'] == nb['atom_type'])

    isomorphisms = list(gm.isomorphisms_iter())
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
                        for i in range(mol_b.GetNumAtoms())]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array([conf_a.GetAtomPosition(i)
                             for i in range(mol_a.GetNumAtoms())])
        conf_b = mol_b.GetConformer()
        coords_b = np.array([conf_b.GetAtomPosition(mapping[i])
                             for i in range(mol_b.GetNumAtoms())])

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)


class AppendVirtualNodes:
    def __init__(self, max_ligand_size, atom_encoder, symbol):
        self.max_ligand_size = max_ligand_size
        self.atom_encoder = atom_encoder
        self.vidx = atom_encoder[symbol]

    def __call__(self, data):

        n_virt = self.max_ligand_size - data['num_lig_atoms']
        mu = data['lig_coords'].mean(0, keepdim=True)
        sigma = data['lig_coords'].std(0).max()
        virt_coords = torch.randn(n_virt, 3) * sigma + mu

        # insert virtual atom column
        one_hot = torch.cat((data['lig_one_hot'][:, :self.vidx],
                            torch.zeros(data['num_lig_atoms'])[:, None],
                            data['lig_one_hot'][:, self.vidx:]), dim=1)
        virt_one_hot = torch.zeros(n_virt, len(self.atom_encoder))
        virt_one_hot[:, self.vidx] = 1
        virt_mask = torch.ones(n_virt) * data['lig_mask'][0]

        data['lig_coords'] = torch.cat((data['lig_coords'], virt_coords))
        data['lig_one_hot'] = torch.cat((one_hot, virt_one_hot))
        data['num_lig_atoms'] = self.max_ligand_size
        data['lig_mask'] = torch.cat((data['lig_mask'], virt_mask))
        data['num_virtual_atoms'] = n_virt

        return data

def sdf_to_smiles(sdf_file):
    """convert SDF file to canonical SMILES."""
    suppl = Chem.SDMolSupplier(str(sdf_file))
    smiles_list = []
    for mol in suppl:
        if mol is not None:
            smi = Chem.MolToSmiles(mol, canonical=True)
            smiles_list.append(smi)
    return smiles_list


# used in training step for timestep permutations

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


# Ensure this ROOT path points to the parent directory containing all the complex subdirectories
ROOT = Path("data/crossdocked_pocket10").resolve()

@functools.lru_cache(maxsize=4096)
def find_gt_files(sample_name: str, root: Path = ROOT):
    """
    Return (ligand.sdf, pocket.pdb) for *sample_name*.
    This version robustly handles various path-like input strings.
    """
    # --- 0) Clean up the input string ---
    name = sample_name.strip()
    name = re.split(r"_pocket10\.pdb", name, 1)[0]
    name = re.sub(r"\.(pdb|sdf)$", "", name)

    # --- 1) Robustly parse the name into subdir and basename ---
    # This is the key part that you might have changed.
    # It treats the cleaned name as a path and extracts the last two components.
    path_parts = Path(name).parts
    
    if len(path_parts) >= 2:
        # Subdir is the second to last part (the parent folder)
        subdir = path_parts[-2]
        # Basename is the very last part (the file name without extension)
        basename = path_parts[-1]
    else:
        # Fallback for simple names without a directory separator
        toks = name.split("_")
        subdir, basename = "_".join(toks[:4]), name

    folder = root / subdir
    if not folder.is_dir():
        raise FileNotFoundError(f"Cross-docked folder not found: {folder}")

    # --- 2) Find the ligand and pocket files ---
    ligand_path = folder / f"{basename}.sdf"
    pocket_path = folder / f"{basename}_pocket10.pdb"

    # Fallback using glob if exact names don't work
    if not ligand_path.exists():
        hits = list(folder.glob(f"{basename}*.sdf"))
        if len(hits) == 1:
            ligand_path = hits[0]
    if not pocket_path.exists():
        hits = list(folder.glob(f"{basename}*_pocket10.pdb"))
        if len(hits) == 1:
            pocket_path = hits[0]

    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand SDF not found for '{sample_name}'")
    if not pocket_path.exists():
        raise FileNotFoundError(f"Pocket PDB not found for '{sample_name}'")

    return ligand_path.resolve(), pocket_path.resolve()


# # This path should still be correct
# ROOT = Path("data/processed_crossdock_noH_full_temp").resolve()

# @functools.lru_cache(maxsize=4096)
# def find_gt_files(sample_name: str, root: Path = ROOT):
#     """
#     Return (ligand.sdf, pocket.pdb) for *sample_name* from a flat directory.
    
#     This version is robust enough to handle concatenated path strings found in the .npz file.
#     Example input: 'data/.../7e2z_E_9SC_pocket_only.pdb_data/.../7e2z_E_9SC.sdf'
#     Desired base_name: '7e2z_E_9SC'
#     """
#     # --- 1) Clean up the sample name to get a base name ---
#     # This is the key part for fixing the issue with your .npz names.
#     # First, split the string on the unique pocket identifier. We take the first part.
#     # '.../7e2z_E_9SC_pocket_only.pdb_...' -> '.../7e2z_E_9SC'
#     try:
#         potential_base = sample_name.split('_pocket_only.pdb')[0]
#     except Exception:
#         # Fallback for unexpected formats
#         potential_base = sample_name

#     # Then, get just the filename part, which strips away the directory path.
#     # 'data/drd2_strucutres/processed_ligand_free_pockets_drd2/7e2z_E_9SC' -> '7e2z_E_9SC'
#     base_name = Path(potential_base).name

#     # --- 2) Construct the expected filenames ---
#     ligand_path = root / f"{base_name}.sdf"
#     pocket_path = root / f"{base_name}_pocket_only.pdb"

#     # --- 3) Verify that the files exist ---
#     if not ligand_path.exists():
#         raise FileNotFoundError(f"Ligand SDF not found for base name '{base_name}' at: {ligand_path}")
#     if not pocket_path.exists():
#         raise FileNotFoundError(f"Pocket PDB not found for base name '{base_name}' at: {pocket_path}")

#     return ligand_path.resolve(), pocket_path.resolve()




def save_generation_triplet(
    *,
    run_root: Path,
    epoch: Optional[int],
    tag: str,
    gen_mol: Chem.Mol,
    ref_mol: Optional[Chem.Mol] = None,
    pocket_xyz: Optional[np.ndarray] = None,
    pocket_structure: Optional[Any] = None,  # Add Bio.PDB structure
    save_ref_and_pocket_once: bool = False,
):
    """
    Save generated ligand, reference ligand (optional) and pocket coords
    to `<run_root>/generation/epoch_XXXX/`.
    """
    epoch_dir = run_root / "generation" / (
        f"epoch_{epoch:04d}" if epoch is not None else "epoch_unknown"
    )
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generated molecule
    gen_path = epoch_dir / f"{tag}_gen.sdf"
    Chem.MolToMolFile(gen_mol, str(gen_path))
    
    # Save reference molecule if provided
    if ref_mol is not None:
        ref_path = epoch_dir / f"{tag}_ref.sdf"
        if (not save_ref_and_pocket_once) or (not ref_path.exists()):
            Chem.MolToMolFile(ref_mol, str(ref_path))
    
    # Save pocket structure
    if pocket_structure is not None:
        # Save the original pocket structure with all atom/residue info
        poc_path = epoch_dir / f"{tag}_pocket.pdb"
        if (not save_ref_and_pocket_once) or (not poc_path.exists()):
            io = PDBIO()
            io.set_structure(pocket_structure)
            io.save(str(poc_path))
    elif pocket_xyz is not None:
        # If we have coordinates but no structure, print warning
        print(f"[WARNING] {tag}: Have pocket coordinates but no structure - pocket not saved")
    else:
        # No pocket data at all
        print(f"[WARNING] {tag}: No pocket data provided - pocket not saved")
        
        

def biopdb_structure_to_rdkit(structure):
    """Convert Bio.PDB structure to RDKit molecule preserving coordinates"""
    try:
        
        io = PDBIO()
        io.set_structure(structure)
        pdb_string = StringIO()
        io.save(pdb_string)
        
        mol = Chem.MolFromPDBBlock(pdb_string.getvalue(), removeHs=False, sanitize=False)
        return mol
        
    except Exception as e:
        print(f"Error converting Bio.PDB to RDKit: {e}")
        return None