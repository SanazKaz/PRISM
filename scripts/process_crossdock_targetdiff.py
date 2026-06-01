"""
process_crossdock_targetdiff.py

Builds a TargetDiff-featurised NPZ dataset from the CrossDocked pocket10
directory, mirroring what process_crossdock.py (DiffSBDD) does but using
TargetDiff's own PDBProtein parser and FeaturizeProteinAtom encoding.

Protein features are 27-dim — identical to what ScorePosNet3D was pretrained
on:
    [0:6]   element one-hot  (H, C, N, O, S, Se)
    [6:26]  amino-acid one-hot (20 standard residues)
    [26]    backbone flag  (1 if atom name in {CA, C, N, O})

Ligand features remain DiffSBDD-style element one-hots so that the reward
pipeline and SMILES computation are unaffected.

Usage
-----
    python -m scripts.process_crossdock_targetdiff \\
        --crossdocked_dir /path/to/crossdocked_pocket10 \\
        --split_path     /path/to/split_by_name.pt \\
        --output_dir     /path/to/output \\
        --dist_cutoff    8.0

The script expects the same directory layout that TargetDiff's
extract_pockets.py produces, i.e. relative paths in split_by_name.pt resolve
under --crossdocked_dir.
"""

import sys
import os
import argparse
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — make both project root and TargetDiff importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT   = Path(__file__).resolve().parents[1]
_TARGETDIFF_ROOT = _PROJECT_ROOT / 'src' / 'models' / 'targetdiff'
_DIFFSBDD_ROOT   = _PROJECT_ROOT / 'src' / 'models' / 'diffsbdd'

# DiffSBDD must be first so its bare `import utils` resolves to
# src/models/diffsbdd/utils.py and not PRISM's utils package.
sys.path.insert(0, str(_DIFFSBDD_ROOT))
sys.path.insert(1, str(_TARGETDIFF_ROOT))
sys.path.insert(2, str(_PROJECT_ROOT))

# DiffSBDD utilities for ligand encoding + SMILES (unchanged from existing pipeline)
from analysis.molecule_builder import build_molecule           # noqa: E402
from analysis.metrics import rdmol_to_smiles                   # noqa: E402
from constants import dataset_params                           # noqa: E402
from rdkit import Chem                                         # noqa: E402
from scipy.ndimage import gaussian_filter                      # noqa: F811


# ---------------------------------------------------------------------------
# TargetDiff protein featurisation
# Implements FeaturizeProteinAtom (transforms.py) on numpy arrays directly
# so we don't need a full ProteinLigandData object.
# ---------------------------------------------------------------------------
_TD_ATOMIC_NUMS = np.array([1, 6, 7, 8, 16, 34], dtype=np.int64)   # H C N O S Se

_PDBProtein = None


def _get_pdb_protein_cls():
    """Load PDBProtein via importlib to avoid the utils namespace collision.

    DiffSBDD ships a flat `utils.py` module, and TargetDiff ships a `utils/`
    package. When DiffSBDD's root is first on sys.path, a normal
    `from utils.data import PDBProtein` fails because Python finds the flat
    module first. importlib.util.spec_from_file_location bypasses sys.path.
    """
    global _PDBProtein
    if _PDBProtein is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            '_prism_targetdiff_data',
            str(_TARGETDIFF_ROOT / 'utils' / 'data.py'),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PDBProtein = mod.PDBProtein
    return _PDBProtein


def featurize_pocket_targetdiff(pdb_path: str) -> tuple:
    """
    Parse a pocket PDB with TargetDiff's own PDBProtein and return
    (coords [N,3], features [N,27]).

    Uses PDBProtein.to_dict_atom() — the exact same function called during
    TargetDiff's training data loading — so the output is guaranteed to match
    what ScorePosNet3D's protein_atom_emb was trained on.
    """
    PDBProtein = _get_pdb_protein_cls()
    protein = PDBProtein(pdb_path)
    d = protein.to_dict_atom()

    element        = np.atleast_1d(d['element'].astype(np.int64))           # [N]
    atom_to_aa     = np.atleast_1d(d['atom_to_aa_type'].astype(np.int64))   # [N]
    is_backbone    = np.atleast_1d(d['is_backbone'].astype(np.float32))     # [N]
    pos            = np.atleast_2d(d['pos'].astype(np.float32))             # [N, 3]

    if pos.shape[0] < 10:
        raise Exception(f"Degenerate pocket: only {pos.shape[0]} atoms parsed — likely a malformed PDB")

    # 6-dim element one-hot
    elem_feat = (element[:, None] == _TD_ATOMIC_NUMS[None, :]).astype(np.float32)

    # 20-dim AA one-hot — clip to [0,19] in case of non-standard residues
    aa_feat = np.eye(20, dtype=np.float32)[np.clip(atom_to_aa, 0, 19)]

    # 1-dim backbone flag
    bb_feat = is_backbone.reshape(-1, 1)

    features = np.concatenate([elem_feat, aa_feat, bb_feat], axis=1)   # [N, 27]
    return pos, features


# ---------------------------------------------------------------------------
# Ligand parsing — DiffSBDD element one-hots (unchanged from existing NPZ)
# ---------------------------------------------------------------------------

def featurize_ligand_diffsbdd(sdf_path: str, atom_dict: dict) -> tuple:
    """
    Returns (coords [M,3], one_hot [M, n_atom_types]) using DiffSBDD's
    element encoding so that reward scoring and SMILES computation stay
    compatible with the existing pipeline.
    """
    try:
        mol = Chem.SDMolSupplier(str(sdf_path))[0]
        if mol is None:
            raise Exception("SDMolSupplier returned None")
    except Exception as e:
        raise Exception(f"Cannot read SDF ({sdf_path}): {e}")

    lig_atoms = [a.GetSymbol() for a in mol.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict or a.GetSymbol() != 'H')]
    if not lig_atoms:
        raise Exception("No valid atoms found in ligand")

    lig_coords = np.array([
        list(mol.GetConformer(0).GetAtomPosition(i))
        for i in range(mol.GetNumAtoms())
    ], dtype=np.float32)

    lig_one_hot = np.stack([
        np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
        for a in lig_atoms
    ]).astype(np.float32)

    return lig_coords, lig_one_hot


# ---------------------------------------------------------------------------
# Statistics helpers (identical to process_crossdock.py)
# ---------------------------------------------------------------------------

def compute_smiles(positions, one_hot, mask, dataset_info):
    print("Computing SMILES...")
    atom_types = np.argmax(one_hot, axis=-1)
    sections   = np.where(np.diff(mask))[0] + 1
    positions  = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []
    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))
    for i, (pos, atype) in pbar:
        mol = build_molecule(pos, atype, dataset_info)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            continue
        smi = rdmol_to_smiles(mol)
        if smi is not None:
            mols_smiles.append(smi)
        pbar.set_description(f'{len(mols_smiles)}/{i+1} successful')
    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    idx_lig, n_lig = np.unique(lig_mask, return_counts=True)
    idx_poc, n_poc = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_poc)

    hist = np.zeros((np.max(n_lig) + 1, np.max(n_poc) + 1))
    for nl, np_ in zip(n_lig, n_poc):
        hist[nl, np_] += 1

    if smooth_sigma is not None:
        hist = gaussian_filter(hist, sigma=smooth_sigma, order=0,
                               mode='constant', cval=0.0, truncate=4.0)
    return hist


def saveall(filename, names, lig_coords, lig_one_hot, lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask):
    np.savez(filename,
             names=np.array(names, dtype=object),
             lig_coords=lig_coords,
             lig_one_hot=lig_one_hot,
             lig_mask=lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    datadir    = Path(args.crossdocked_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DiffSBDD dataset info for ligand encoding + SMILES
    dataset_info = dataset_params['crossdock_full']
    atom_dict    = dataset_info['atom_encoder']

    # Load split — same split_by_name.pt used by both DiffSBDD and TargetDiff
    print(f"Loading split from {args.split_path}")
    data_split = torch.load(args.split_path)

    # Create a validation set (DiffSBDD used 300 random train samples)
    data_split['val'] = random.sample(data_split['train'], 300)

    print(f"Split sizes — train: {len(data_split['train'])}  "
          f"val: {len(data_split['val'])}  test: {len(data_split['test'])}")

    n_samples_after = {}
    failed_all = []

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} ({len(data_split[split])} pairs)")
        print('='*60)

        lig_coords_list, lig_one_hot_list, lig_mask_list   = [], [], []
        poc_coords_list, poc_one_hot_list, poc_mask_list   = [], [], []
        names = []
        count = 0
        num_failed = 0
        tic = time()

        pbar = tqdm(data_split[split], desc=f'{split} #failed=0')
        for pocket_fn, ligand_fn in pbar:
            pdb_path = str(datadir / pocket_fn)
            sdf_path = str(datadir / ligand_fn)

            try:
                poc_coords, poc_one_hot = featurize_pocket_targetdiff(pdb_path)
                lig_coords, lig_one_hot = featurize_ligand_diffsbdd(sdf_path, atom_dict)
            except Exception as e:
                num_failed += 1
                failed_all.append((pocket_fn, ligand_fn, str(e)))
                pbar.set_description(f'{split} #failed={num_failed}')
                continue

            names.append(f'{pocket_fn}_{ligand_fn}')
            lig_coords_list.append(lig_coords)
            lig_one_hot_list.append(lig_one_hot)
            lig_mask_list.append(count * np.ones(len(lig_coords)))
            poc_coords_list.append(poc_coords)
            poc_one_hot_list.append(poc_one_hot)
            poc_mask_list.append(count * np.ones(len(poc_coords)))
            count += 1

        print(f"Processed {count} pairs in {(time()-tic)/60:.1f} min  "
              f"({num_failed} failed)")

        if not names:
            print(f"[ERROR] No samples for {split} — skipping NPZ save.")
            n_samples_after[split] = 0
            continue

        lig_coords_np  = np.concatenate(lig_coords_list,  axis=0)
        lig_one_hot_np = np.concatenate(lig_one_hot_list, axis=0)
        lig_mask_np    = np.concatenate(lig_mask_list,    axis=0)
        poc_coords_np  = np.concatenate(poc_coords_list,  axis=0)
        poc_one_hot_np = np.concatenate(poc_one_hot_list, axis=0)
        poc_mask_np    = np.concatenate(poc_mask_list,    axis=0)

        out_npz = output_dir / f'{split}.npz'
        saveall(out_npz, names,
                lig_coords_np, lig_one_hot_np, lig_mask_np,
                poc_coords_np, poc_one_hot_np, poc_mask_np)
        print(f"Saved {out_npz}  ({count} samples, pocket_one_hot shape: {poc_one_hot_np.shape})")
        n_samples_after[split] = count

    # --- Statistics from training set ---
    train_npz = output_dir / 'train.npz'
    if train_npz.exists() and n_samples_after.get('train', 0) > 0:
        print("\nComputing training-set statistics...")
        with np.load(train_npz, allow_pickle=True) as d:
            lig_coords   = d['lig_coords']
            lig_one_hot  = d['lig_one_hot']
            lig_mask     = d['lig_mask']
            pocket_mask  = d['pocket_mask']

        train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask, dataset_info)
        np.save(output_dir / 'train_smiles.npy', train_smiles)
        print(f"Saved train_smiles.npy ({len(train_smiles)} SMILES)")

        n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
        np.save(output_dir / 'size_distribution.npy', n_nodes)
        print(f"Saved size_distribution.npy  shape={n_nodes.shape}")

    summary = (
        f"# TargetDiff CrossDocked NPZ — 27-dim protein features\n\n"
        f"num_samples train: {n_samples_after.get('train', 0)}\n"
        f"num_samples val:   {n_samples_after.get('val',   0)}\n"
        f"num_samples test:  {n_samples_after.get('test',  0)}\n\n"
        f"pocket_one_hot dim: 27  "
        f"(6 element + 20 AA + 1 backbone, FeaturizeProteinAtom)\n"
        f"lig_one_hot dim:    {len(atom_dict)}  (DiffSBDD crossdock_full encoder)\n\n"
        f"num_failed: {len(failed_all)}\n"
    )
    (output_dir / 'summary.txt').write_text(summary)
    print(f"\n{summary}")

    if failed_all:
        print(f"Failed pairs ({len(failed_all)}):")
        for p, l, e in failed_all[:20]:
            print(f"  {p} | {l} | {e}")
        if len(failed_all) > 20:
            print(f"  ... and {len(failed_all)-20} more")

    print(f"\nDone. Output: {output_dir.resolve()}")
    print(f"Point targetdiff_ppo.yaml datadir at: {output_dir.resolve()}")


def smoke_test(args):
    """
    Run featurization on the first 2 train pairs and verify shapes.
    Catches dimension bugs before a full multi-hour run.
    Remove this function (and the --smoke_test flag) once confirmed working.
    """
    datadir = Path(args.crossdocked_dir)
    dataset_info = dataset_params['crossdock_full']
    atom_dict    = dataset_info['atom_encoder']

    print("Loading split...")
    data_split = torch.load(args.split_path)
    pairs = data_split['train'][:2]

    print(f"Smoke-testing on {len(pairs)} pairs:")
    poc_coords_list, poc_one_hot_list = [], []
    lig_coords_list, lig_one_hot_list = [], []

    for i, (pocket_fn, ligand_fn) in enumerate(pairs):
        pdb_path = str(datadir / pocket_fn)
        sdf_path = str(datadir / ligand_fn)
        print(f"\n  [{i}] pocket: {pocket_fn}")
        print(f"      ligand: {ligand_fn}")

        poc_coords, poc_one_hot = featurize_pocket_targetdiff(pdb_path)
        lig_coords, lig_one_hot = featurize_ligand_diffsbdd(sdf_path, atom_dict)

        print(f"      poc_coords  shape={poc_coords.shape}  dtype={poc_coords.dtype}")
        print(f"      poc_one_hot shape={poc_one_hot.shape}  dtype={poc_one_hot.dtype}")
        print(f"      lig_coords  shape={lig_coords.shape}  dtype={lig_coords.dtype}")
        print(f"      lig_one_hot shape={lig_one_hot.shape}  dtype={lig_one_hot.dtype}")

        assert poc_coords.ndim == 2 and poc_coords.shape[1] == 3, \
            f"poc_coords must be [N,3], got {poc_coords.shape}"
        assert poc_one_hot.ndim == 2 and poc_one_hot.shape[1] == 27, \
            f"poc_one_hot must be [N,27], got {poc_one_hot.shape}"
        assert lig_coords.ndim == 2 and lig_coords.shape[1] == 3, \
            f"lig_coords must be [M,3], got {lig_coords.shape}"
        assert poc_coords.shape[0] == poc_one_hot.shape[0], \
            "poc_coords and poc_one_hot atom count mismatch"

        poc_coords_list.append(poc_coords)
        poc_one_hot_list.append(poc_one_hot)
        lig_coords_list.append(lig_coords)
        lig_one_hot_list.append(lig_one_hot)

    # Verify concatenation — this is exactly what main() does before saving
    print("\nVerifying np.concatenate across pairs...")
    poc_coords_np  = np.concatenate(poc_coords_list,  axis=0)
    poc_one_hot_np = np.concatenate(poc_one_hot_list, axis=0)
    lig_coords_np  = np.concatenate(lig_coords_list,  axis=0)
    lig_one_hot_np = np.concatenate(lig_one_hot_list, axis=0)

    print(f"  poc_coords_np  : {poc_coords_np.shape}")
    print(f"  poc_one_hot_np : {poc_one_hot_np.shape}")
    print(f"  lig_coords_np  : {lig_coords_np.shape}")
    print(f"  lig_one_hot_np : {lig_one_hot_np.shape}")

    print("\nSmoke test PASSED — shapes are correct, concatenation works.")
    print("You can now submit the full job.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build TargetDiff-featurised NPZ from CrossDocked pocket10.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--crossdocked_dir', type=str, required=True,
                        help='Path to crossdocked_pocket10/ directory')
    parser.add_argument('--split_path', type=str, required=True,
                        help='Path to split_by_name.pt')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for val split sampling')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run on 2 pairs only to verify shapes before a full run')
    args = parser.parse_args()

    random.seed(args.seed)
    if args.smoke_test:
        smoke_test(args)
    else:
        main(args)
