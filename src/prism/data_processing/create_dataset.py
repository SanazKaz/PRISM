#!/usr/bin/env python

"""
Processes pairs of (ligand-free pocket PDB, ligand SDF) files into
numpy .npz files for use in machine learning models.

This script reads lists of file basenames from a split file
(e.g., train_data.txt) to build train/val/test datasets.

It includes an optional de-duplication step to process only one
ligand instance per PDB/ligand combination.

Adapted from DiffSBDD/process_crossdock.py
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from time import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1_extended, is_aa

from rdkit import Chem

# ---------------------------------------------------------------------------
# TargetDiff protein featurisation constants
# Matches FeaturizeProteinAtom in src/models/targetdiff/utils/transforms.py
# exactly — do not change without updating the model config.
# ---------------------------------------------------------------------------
_TD_ELEMENT_TO_ATOMIC_NUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'SE': 34}
_TD_ATOMIC_NUMS = [1, 6, 7, 8, 16, 34]   # 6-dim element one-hot
_TD_AA_ORDER = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS',
    'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN',
    'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
]                                          # 20-dim AA one-hot
_TD_AA_INDEX = {aa: i for i, aa in enumerate(_TD_AA_ORDER)}
_TD_BACKBONE = {'CA', 'C', 'N', 'O'}      # 1-dim backbone flag
# Total: 6 + 20 + 1 = 27 dims

project_root = os.getcwd()
diffsbdd_path = os.path.join(project_root, 'src', 'models', 'diffsbdd')
if diffsbdd_path not in sys.path:
    sys.path.append(diffsbdd_path)

from src.models.diffsbdd.analysis.molecule_builder import build_molecule
from src.models.diffsbdd.analysis.metrics import rdmol_to_smiles
from src.models.diffsbdd.constants import covalent_radii, dataset_params


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_split_file(filepath, deduplicate=False):
    """
    Reads a split file, returning a list of clean basenames.

    Args:
        filepath: Path to the .txt file.
        deduplicate: If True, keeps only the first instance for each
                     PDB-ligand pair (e.g., keeps '7e2z_9SC_A_501' and
                     discards '7e2z_9SC_B_501').
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Split file not found: {str(filepath)}")

    basenames = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                basenames.append(line)

    if not basenames:
        print(f"Warning: Split file '{filepath}' was empty or only contained comments.")
        return []

    if not deduplicate:
        return basenames

    print(f"  De-duplicating {len(basenames)} entries...")
    final_basenames = []
    seen_keys = set()
    for basename in basenames:
        parts = basename.split('_')
        if len(parts) < 2:
            print(f"    Skipping malformed basename: {basename}")
            continue

        # Key is PDB_LIG (e.g. "7e2z_9SC"); ignores chain/residue variants
        key = f"{parts[0]}_{parts[1]}"

        if key not in seen_keys:
            final_basenames.append(basename)
            seen_keys.add(key)

    print(f"  Returning {len(final_basenames)} unique entries.")
    return final_basenames


def robust_read_sdf(sdf_path):
    """Robust SDF reading that tries multiple RDKit methods."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*molecule is tagged as 2D.*")

        try:
            mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=True)
            if mol is not None:
                return mol
        except: pass

        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            mol = next(iter(supplier), None)
            if mol is not None:
                return mol
        except: pass

        raise Exception(f"cannot read sdf mol ({sdf_path})")


def process_ligand_and_pocket(pdbfile, sdffile, atom_dict, aa_encoder, dist_cutoff):
    """
    Processes a single PDB/SDF pair into coordinates and one-hot encodings.
    """
    try:
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
    except Exception as e:
        raise Exception(f"PDB structure parsing failed: {str(e)}")

    ligand = robust_read_sdf(sdffile)
    if ligand is None:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    try:
        lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                    if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H')]
        if not lig_atoms:
            raise Exception("No valid atoms found in ligand")

        # Filter coords identically to lig_atoms: exclude H atoms not in atom_dict.
        # Using range(GetNumAtoms()) would include explicit H atoms and cause a
        # coord/one_hot shape mismatch when the SDF has explicit H positions.
        lig_coords = np.array([
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx, a in enumerate(ligand.GetAtoms())
            if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H')
        ])
    except Exception as e:
        raise Exception(f"ligand atom processing failed:{str(e)}")

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
            for a in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(f'Atom {e} not in atom dict ({sdffile})')

    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(
                    -1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }

    full_atoms = np.concatenate(
        [np.array([atom.element for atom in res.get_atoms()])
         for res in pocket_residues], axis=0)
    full_coords = np.concatenate(
        [np.array([atom.coord for atom in res.get_atoms()])
         for res in pocket_residues], axis=0)
    try:
        pocket_one_hot = []
        for a in full_atoms:
            if a in atom_dict:
                atom = np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
            elif a not in ('H', 'D'):  # treat deuterium like hydrogen
                atom = np.eye(1, len(atom_dict), len(atom_dict) - 1).squeeze()
            pocket_one_hot.append(atom)
        pocket_one_hot = np.stack(pocket_one_hot)
    except KeyError as e:
        raise KeyError(f'{e} not in atom dict ({pdbfile})')

    pocket_data = {
        'pocket_coords': full_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids
    }

    return ligand_data, pocket_data


def compute_smiles(positions, one_hot, mask, dataset_info):
    """
    Computes SMILES with robust error handling to prevent pipeline crashes.
    """
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)
    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []
    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))

    for i, (pos, atom_type) in pbar:
        try:
            mol = build_molecule(pos, atom_type, dataset_info)

            if mol is None:
                continue

            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            mol = rdmol_to_smiles(mol)
            if mol is not None:
                mols_smiles.append(mol)

            pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

        except Exception:
            continue

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    """
    Compute joint distribution of ligand and pocket node counts.
    """
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    if smooth_sigma is not None:
        joint_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(joint_histogram)}/'
              f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    return joint_histogram


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    """
    Computes histograms of atom and amino acid types.
    """
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1
    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1
    return atom_counts, aa_counts


def saveall(filename, pdb_and_mol_ids, lig_coords, lig_one_hot, lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask):
    np.savez(filename,
             names=np.array(pdb_and_mol_ids, dtype=object),
             lig_coords=lig_coords,
             lig_one_hot=lig_one_hot,
             lig_mask=lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask)
    return True


def _targetdiff_atom_features(atom_element: str, res_name: str, atom_name: str) -> np.ndarray:
    """27-dim feature vector matching TargetDiff's FeaturizeProteinAtom exactly."""
    atomic_num = _TD_ELEMENT_TO_ATOMIC_NUM.get(atom_element.upper(), 0)
    element_vec = [int(atomic_num == n) for n in _TD_ATOMIC_NUMS]      # 6-dim

    aa_vec = [0] * 20                                                    # 20-dim
    aa_idx = _TD_AA_INDEX.get(res_name)
    if aa_idx is not None:
        aa_vec[aa_idx] = 1

    backbone_flag = [int(atom_name.strip() in _TD_BACKBONE)]            # 1-dim

    return np.array(element_vec + aa_vec + backbone_flag, dtype=np.float32)


def process_ligand_and_pocket_targetdiff(pdbfile, sdffile, dist_cutoff, atom_dict):
    """
    Like process_ligand_and_pocket but produces 27-dim TargetDiff protein
    features instead of DiffSBDD element one-hots.

    Ligand features use the same DiffSBDD atom_dict encoding as the DiffSBDD
    path so that compute_smiles and size_distribution statistics are consistent.
    The TargetDiff model never consumes ligand features from the NPZ directly.
    """
    try:
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
    except Exception as e:
        raise Exception(f"PDB structure parsing failed: {str(e)}")

    ligand = robust_read_sdf(sdffile)
    if ligand is None:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    lig_coords = np.array([
        list(ligand.GetConformer(0).GetAtomPosition(idx))
        for idx in range(ligand.GetNumAtoms())
        if ligand.GetAtomWithIdx(idx).GetSymbol() != 'H'
    ])
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms() if a.GetSymbol() != 'H']
    if not lig_atoms:
        raise Exception("No valid heavy atoms found in ligand")

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
            for a in lig_atoms
        ]).astype(np.float32)
    except KeyError as e:
        raise KeyError(f'Atom {e} not in atom dict ({sdffile})')

    # Pocket residues within dist_cutoff of any ligand heavy atom
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]

    pocket_feats, pocket_coords_list = [], []
    for res in pocket_residues:
        res_name = res.get_resname()
        for atom in res.get_atoms():
            # Skip hydrogen and deuterium (D appears in neutron/deuterated
            # structures, e.g. CA-II 6bc9/6bbs/4g0c). D is not a recognised
            # element, so it would produce an all-zero element block and fall
            # through to the amino-acid indices, crashing get_type_histograms.
            if atom.element in ('H', 'D'):
                continue
            feat = _targetdiff_atom_features(atom.element, res_name, atom.get_name())
            pocket_feats.append(feat)
            pocket_coords_list.append(atom.get_coord())

    if not pocket_feats:
        raise Exception(f"No pocket atoms found within {dist_cutoff}Å of ligand ({pdbfile})")

    pocket_one_hot = np.stack(pocket_feats)       # [N_pocket, 27]
    pocket_coords  = np.array(pocket_coords_list) # [N_pocket, 3]

    ligand_data = {'lig_coords': lig_coords, 'lig_one_hot': lig_one_hot}
    pocket_data = {'pocket_coords': pocket_coords,
                   'pocket_one_hot': pocket_one_hot,
                   'pocket_ids': pocket_ids}
    return ligand_data, pocket_data


# =============================================================================
# MAIN SCRIPT LOGIC
# =============================================================================

def main(args):
    model = getattr(args, 'model', 'diffsbdd')

    base_input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pocket_dir = base_input_dir / "pocket_files"
    sdf_dir = base_input_dir / "sdf_files"

    print(f"Starting dataset creation.")
    print(f"Base Input Directory: {base_input_dir.resolve()}")
    print(f"  Reading Pockets from: {pocket_dir.resolve()}")
    print(f"  Reading SDFs from:   {sdf_dir.resolve()}")
    print(f"Output (NPZ) Directory: {output_dir.resolve()}")

    output_dir.mkdir(exist_ok=True, parents=True)

    if not pocket_dir.exists():
        print(f"Error: Pocket directory not found: {pocket_dir}", file=sys.stderr)
        sys.exit(1)
    if not sdf_dir.exists():
        print(f"Error: SDF directory not found: {sdf_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        dataset_info = dataset_params[args.dataset_info_key]
        print(f"Using dataset info key: '{args.dataset_info_key}'")
        amino_acid_dict = dataset_info['aa_encoder']
        atom_dict = dataset_info['atom_encoder']
    except KeyError:
        print(f"Error: '{args.dataset_info_key}' not found in dataset_params from constants.py", file=sys.stderr)
        sys.exit(1)

    dist_cutoff = args.dist_cutoff
    dataset_name = args.dataset_name

    split_file_path = base_input_dir / args.split_file

    splits_to_read = {
        'train': split_file_path,
        'val':   split_file_path,
        'test':  split_file_path,
    }

    print("Reading basenames from split files...")
    splits_with_basenames = {}
    try:
        for split_name, filepath in splits_to_read.items():
            print(f"Processing split: {split_name} (from {filepath.name})")
            splits_with_basenames[split_name] = read_split_file(filepath, deduplicate=args.deduplicate)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    files_for_sets = {}
    for split, basenames in splits_with_basenames.items():
        if not basenames:
            print(f"Skipping '{split}' set as no basenames were found in its split file.")
            continue

        files_for_sets[split] = [
            (
                pocket_dir / f'{name}_pocket.pdb',
                sdf_dir / f'{name}.sdf'
            )
            for name in basenames
        ]
        print(f"Found {len(files_for_sets[split])} file pairs for '{split}' split.")

    n_samples_after = {}

    for split_name, files_to_process in files_for_sets.items():
        if not files_to_process:
            print(f"\n--- Skipping {split_name.upper()} set (no files) ---")
            continue

        print(f"\n--- Processing {split_name.upper()} set ({len(files_to_process)} pairs) ---")

        lig_coords, lig_one_hot, lig_mask = [], [], []
        pocket_coords, pocket_one_hot, pocket_mask = [], [], []
        pdb_and_mol_ids = []
        failed_save = []
        count = 0
        tic = time()

        for pocket_fn, ligand_fn in files_to_process:
            if not pocket_fn.exists():
                print(f"Warning: Pocket file not found, skipping: {pocket_fn.name}")
                failed_save.append((pocket_fn, ligand_fn, "Pocket PDB not found"))
                continue
            if not ligand_fn.exists():
                print(f"Warning: Ligand file not found, skipping: {ligand_fn.name}")
                failed_save.append((pocket_fn, ligand_fn, "Ligand SDF not found"))
                continue

            try:
                if model == 'targetdiff':
                    ligand_data, pocket_data = process_ligand_and_pocket_targetdiff(
                        str(pocket_fn), str(ligand_fn),
                        dist_cutoff=dist_cutoff,
                        atom_dict=atom_dict,
                    )
                else:
                    ligand_data, pocket_data = process_ligand_and_pocket(
                        str(pocket_fn), str(ligand_fn),
                        atom_dict=atom_dict,
                        aa_encoder=amino_acid_dict,
                        dist_cutoff=dist_cutoff,
                    )

                pdb_and_mol_ids.append(f"{pocket_fn.name}_{ligand_fn.name}")
                lig_coords.append(ligand_data['lig_coords'])
                lig_one_hot.append(ligand_data['lig_one_hot'])
                lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))
                pocket_coords.append(pocket_data['pocket_coords'])
                pocket_one_hot.append(pocket_data['pocket_one_hot'])
                pocket_mask.append(count * np.ones(len(pocket_data['pocket_coords'])))
                count += 1

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"  [FAIL] {pocket_fn.name} / {ligand_fn.name}\n    Reason: {error_msg}")
                failed_save.append((pocket_fn.name, ligand_fn.name, error_msg))
                continue

        print(f"Processing {split_name} set took {(time() - tic) / 60.0:.2f} minutes.")
        if failed_save:
            print(f"--- Failed Files for {split_name.upper()} ---")
            for p, l, e in failed_save:
                print(f"  - PDB: {p}, SDF: {l}\n    Reason: {e}")

        if pdb_and_mol_ids:
            lig_coords = np.concatenate(lig_coords, axis=0)
            lig_one_hot = np.concatenate(lig_one_hot, axis=0)
            lig_mask = np.concatenate(lig_mask, axis=0)
            pocket_coords = np.concatenate(pocket_coords, axis=0)
            pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
            pocket_mask = np.concatenate(pocket_mask, axis=0)

            output_npz_file = output_dir / f'{split_name}.npz'
            print(f"--- Saving {split_name} dataset to {output_npz_file} ---")
            saveall(output_npz_file, pdb_and_mol_ids, lig_coords,
                    lig_one_hot, lig_mask, pocket_coords,
                    pocket_one_hot, pocket_mask)
            print(f"Success: Created {split_name}.npz with {len(pdb_and_mol_ids)} samples.")
            n_samples_after[split_name] = len(pdb_and_mol_ids)
        else:
            print(f"Error: No samples were processed successfully for {split_name}.", file=sys.stderr)
            n_samples_after[split_name] = 0

    train_npz_path = output_dir / 'train.npz'
    if train_npz_path.exists() and n_samples_after.get('train', 0) > 0:
        print("\n--- Computing Statistics for the TRAINING dataset ---")
        with np.load(train_npz_path, allow_pickle=True) as data:
            train_lig_coords = data['lig_coords']
            train_lig_one_hot = data['lig_one_hot']
            train_lig_mask = data['lig_mask']
            train_pocket_one_hot = data['pocket_one_hot']
            train_pocket_mask = data['pocket_mask']

        train_smiles = compute_smiles(train_lig_coords, train_lig_one_hot, train_lig_mask, dataset_info)
        np.save(output_dir / 'train_smiles.npy', train_smiles)

        n_nodes = get_n_nodes(train_lig_mask, train_pocket_mask, smooth_sigma=1.0)
        np.save(output_dir / 'size_distribution.npy', n_nodes)

        atom_hist, aa_hist = get_type_histograms(train_lig_one_hot, train_pocket_one_hot, atom_dict, amino_acid_dict)

        summary_string = f"""# SUMMARY for {dataset_name.upper()} Dataset Split

# After processing
num_samples train: {n_samples_after.get('train', 0)}
num_samples val:   {n_samples_after.get('val', 0)}
num_samples test:  {n_samples_after.get('test', 0)}

# Info (from training set)
'atom_encoder': {atom_dict}
'atom_decoder': {list(atom_dict.keys())}
'aa_encoder': {amino_acid_dict}
'aa_decoder': {list(amino_acid_dict.keys())}
'atom_hist': {atom_hist}
'aa_hist': {aa_hist}
'n_nodes_shape': {n_nodes.shape}
"""
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_string)
        print(f"\nSuccess: Summary and stats saved in: {output_dir}")
    else:
        print("\nError: Training set did not process correctly. Cannot compute statistics.", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process pocket/ligand pairs into .npz datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Input directory from preprocess_data.py (contains pocket_files/ and sdf_files/).')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory to save the final .npz files and statistics.')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Filename of the split file located in input_dir (e.g. "train_data.txt").')
    parser.add_argument('--keep_duplicates', action='store_false', dest='deduplicate',
                        help='Keep all ligand instances per PDB-ligand ID pair (deduplication is on by default).')
    parser.add_argument('--dataset_name', type=str, default='data',
                        help='Label used in the summary file.')
    parser.add_argument('--dataset_info_key', type=str, default='crossdock_full',
                        help='Key from constants.py dataset_params for atom/AA encoders.')
    parser.add_argument('--dist_cutoff', type=float, default=5.0,
                        help='Distance cutoff (Å) for defining pocket residues.')
    parser.add_argument('--model', choices=['diffsbdd', 'targetdiff'], default='diffsbdd',
                        help='Pocket featurisation format: diffsbdd (10-dim element one-hots) '
                             'or targetdiff (27-dim element + AA + backbone features).')

    args = parser.parse_args()
    args.deduplicate = True  # default on when called standalone; --keep_duplicates flips it
    main(args)
