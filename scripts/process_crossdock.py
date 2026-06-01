"""
process_crossdock.py — Build an NPZ dataset from CrossDocked pocket10.

Supports both DiffSBDD (10-dim element one-hots) and TargetDiff (27-dim
element + AA + backbone features) via the --model flag, so a single script
replaces the old vendored call and process_crossdock_targetdiff.py.

Adding a new model
------------------
1. Implement a featurize_pocket_<model>(pdb_path) → (coords, features) function.
2. Add an `elif args.model == '<model>':` branch in the main processing loop.
3. Add the model name to the --model choices list.

Usage
-----
    python -m scripts.process_crossdock \\
        --crossdocked_dir /path/to/crossdocked_pocket10 \\
        --split_path      /path/to/split_by_name.pt \\
        --output_dir      /path/to/output \\
        --model           diffsbdd        # or targetdiff
        --dist_cutoff     8.0
        --smoke_test                      # optional: test 2 pairs before full run
"""

import sys
import argparse
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT    = Path(__file__).resolve().parents[1]
_DIFFSBDD_ROOT   = _PROJECT_ROOT / 'src' / 'models' / 'diffsbdd'
_TARGETDIFF_ROOT = _PROJECT_ROOT / 'src' / 'models' / 'targetdiff'

# DiffSBDD must be first so its bare `import utils` resolves to
# src/models/diffsbdd/utils.py and not PRISM's utils package.
sys.path.insert(0, str(_DIFFSBDD_ROOT))
sys.path.insert(1, str(_TARGETDIFF_ROOT))
sys.path.insert(2, str(_PROJECT_ROOT))

# DiffSBDD vendored: import only the pocket+ligand featurizer — the one
# function complex enough that we don't want to duplicate it.
from process_crossdock import process_ligand_and_pocket    # noqa: E402
from analysis.molecule_builder import build_molecule        # noqa: E402
from analysis.metrics import rdmol_to_smiles                # noqa: E402
from constants import dataset_params                        # noqa: E402

from rdkit import Chem                                      # noqa: E402


# ---------------------------------------------------------------------------
# TargetDiff pocket featurisation (27-dim)
# Mirrors FeaturizeProteinAtom from src/models/targetdiff/utils/transforms.py
# using PDBProtein.to_dict_atom() so output is guaranteed to match the
# pretrained ScorePosNet3D checkpoint.
# ---------------------------------------------------------------------------
_TD_ATOMIC_NUMS = np.array([1, 6, 7, 8, 16, 34], dtype=np.int64)  # H C N O S Se


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
    """Parse a pocket PDB with TargetDiff's PDBProtein → (coords [N,3], features [N,27])."""
    PDBProtein = _get_pdb_protein_cls()
    protein = PDBProtein(pdb_path)
    d = protein.to_dict_atom()

    element     = np.atleast_1d(d['element'].astype(np.int64))
    atom_to_aa  = np.atleast_1d(d['atom_to_aa_type'].astype(np.int64))
    is_backbone = np.atleast_1d(d['is_backbone'].astype(np.float32))
    pos         = np.atleast_2d(d['pos'].astype(np.float32))

    if pos.shape[0] < 10:
        raise Exception(f"Degenerate pocket: only {pos.shape[0]} atoms — likely malformed PDB")

    elem_feat = (element[:, None] == _TD_ATOMIC_NUMS[None, :]).astype(np.float32)  # [N, 6]
    aa_feat   = np.eye(20, dtype=np.float32)[np.clip(atom_to_aa, 0, 19)]           # [N, 20]
    bb_feat   = is_backbone.reshape(-1, 1)                                          # [N, 1]

    return pos, np.concatenate([elem_feat, aa_feat, bb_feat], axis=1)


# ---------------------------------------------------------------------------
# Shared ligand featurisation (DiffSBDD-compatible, used by all models)
# Ligand one-hots are stored in the NPZ for size distribution and SMILES
# statistics only; they are never consumed by the TargetDiff model itself.
# ---------------------------------------------------------------------------

def featurize_ligand(sdf_path: str, atom_dict: dict) -> tuple:
    """DiffSBDD-compatible element one-hots for a ligand SDF → (coords, one_hot)."""
    mol = Chem.SDMolSupplier(str(sdf_path))[0]
    if mol is None:
        raise Exception(f"SDMolSupplier returned None ({sdf_path})")

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
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_smiles(positions, one_hot, mask, dataset_info) -> list:
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


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    atom_decoder = list(atom_encoder.keys())
    atom_counts  = {k: 0 for k in atom_encoder}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts  = {k: 0 for k in aa_encoder}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


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
# Processing loop
# ---------------------------------------------------------------------------

def _process_split(split_pairs, datadir, atom_dict, args):
    """
    Featurise all (pocket_fn, ligand_fn) pairs for one split.

    Returns (names, lig_coords, lig_one_hot, lig_mask,
             poc_coords, poc_one_hot, poc_mask, failed_list).
    """
    lig_coords_list, lig_one_hot_list, lig_mask_list = [], [], []
    poc_coords_list, poc_one_hot_list, poc_mask_list = [], [], []
    names = []
    count = num_failed = 0
    failed = []
    tic = time()

    pbar = tqdm(split_pairs, desc=f'#failed=0')
    for pocket_fn, ligand_fn in pbar:
        pdb_path = str(datadir / pocket_fn)
        sdf_path = str(datadir / ligand_fn)

        try:
            if args.model == 'diffsbdd':
                lig_data, poc_data = process_ligand_and_pocket(
                    pdb_path, sdf_path,
                    atom_dict=atom_dict,
                    dist_cutoff=args.dist_cutoff,
                    ca_only=False,
                )
                lig_coords  = lig_data['lig_coords']
                lig_one_hot = lig_data['lig_one_hot']
                poc_coords  = poc_data['pocket_coords']
                poc_one_hot = poc_data['pocket_one_hot']

            elif args.model == 'targetdiff':
                poc_coords, poc_one_hot = featurize_pocket_targetdiff(pdb_path)
                lig_coords, lig_one_hot = featurize_ligand(sdf_path, atom_dict)

            # Add elif for future models here

        except Exception as e:
            num_failed += 1
            failed.append((pocket_fn, ligand_fn, str(e)))
            pbar.set_description(f'#failed={num_failed}')
            continue

        names.append(f'{pocket_fn}_{ligand_fn}')
        lig_coords_list.append(lig_coords)
        lig_one_hot_list.append(lig_one_hot)
        lig_mask_list.append(count * np.ones(len(lig_coords)))
        poc_coords_list.append(poc_coords)
        poc_one_hot_list.append(poc_one_hot)
        poc_mask_list.append(count * np.ones(len(poc_coords)))
        count += 1

    print(f"  {count} pairs processed in {(time()-tic)/60:.1f} min  ({num_failed} failed)")

    if not names:
        return names, None, None, None, None, None, None, failed

    return (
        names,
        np.concatenate(lig_coords_list),
        np.concatenate(lig_one_hot_list),
        np.concatenate(lig_mask_list),
        np.concatenate(poc_coords_list),
        np.concatenate(poc_one_hot_list),
        np.concatenate(poc_mask_list),
        failed,
    )


def main(args):
    datadir    = Path(args.crossdocked_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info    = dataset_params['crossdock_full']
    atom_dict       = dataset_info['atom_encoder']
    amino_acid_dict = dataset_info['aa_encoder']

    print(f"Loading split from {args.split_path}")
    data_split = torch.load(args.split_path)
    data_split['val'] = random.sample(data_split['train'], 300)

    print(f"Split — train: {len(data_split['train'])}  "
          f"val: {len(data_split['val'])}  test: {len(data_split['test'])}")

    n_samples_after = {}
    failed_all = []

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}\n{split.upper()}  ({len(data_split[split])} pairs)\n{'='*60}")

        names, lc, lh, lm, pc, ph, pm, failed = _process_split(
            data_split[split], datadir, atom_dict, args)
        failed_all.extend(failed)

        if not names:
            print(f"[ERROR] No samples for {split} — skipping NPZ save.")
            n_samples_after[split] = 0
            continue

        out_npz = output_dir / f'{split}.npz'
        saveall(out_npz, names, lc, lh, lm, pc, ph, pm)
        print(f"Saved {out_npz}  ({len(names)} samples, pocket_one_hot: {ph.shape})")
        n_samples_after[split] = len(names)

    # --- Statistics from training set ---
    train_npz = output_dir / 'train.npz'
    if train_npz.exists() and n_samples_after.get('train', 0) > 0:
        print("\nComputing training-set statistics...")
        with np.load(train_npz, allow_pickle=True) as d:
            lc, lh, lm, ph, pm = (d['lig_coords'], d['lig_one_hot'], d['lig_mask'],
                                   d['pocket_one_hot'], d['pocket_mask'])

        train_smiles = compute_smiles(lc, lh, lm, dataset_info)
        np.save(output_dir / 'train_smiles.npy', train_smiles)
        print(f"Saved train_smiles.npy  ({len(train_smiles)} SMILES)")

        n_nodes = get_n_nodes(lm, pm, smooth_sigma=1.0)
        np.save(output_dir / 'size_distribution.npy', n_nodes)
        print(f"Saved size_distribution.npy  shape={n_nodes.shape}")

        atom_hist, aa_hist = get_type_histograms(lh, ph, atom_dict, amino_acid_dict)

    model_tag = args.model
    pocket_dim = 27 if args.model == 'targetdiff' else len(atom_dict)
    summary = (
        f"# CrossDocked NPZ — model={model_tag}\n\n"
        f"num_samples train: {n_samples_after.get('train', 0)}\n"
        f"num_samples val:   {n_samples_after.get('val',   0)}\n"
        f"num_samples test:  {n_samples_after.get('test',  0)}\n\n"
        f"pocket_one_hot dim: {pocket_dim}\n"
        f"lig_one_hot dim:    {len(atom_dict)}  (crossdock_full encoder)\n\n"
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

    print(f"Done. Output: {output_dir.resolve()}")
    print(f"Set 'datadir' in your config to: {output_dir.resolve()}")


def smoke_test(args):
    """Run featurisation on 2 train pairs to verify shapes before a full run."""
    datadir      = Path(args.crossdocked_dir)
    dataset_info = dataset_params['crossdock_full']
    atom_dict    = dataset_info['atom_encoder']

    print(f"Loading split from {args.split_path}")
    data_split = torch.load(args.split_path)
    pairs = data_split['train'][:2]

    print(f"Smoke-testing {args.model} on {len(pairs)} pairs:")
    poc_list, lig_list = [], []

    for i, (pocket_fn, ligand_fn) in enumerate(pairs):
        pdb_path = str(datadir / pocket_fn)
        sdf_path = str(datadir / ligand_fn)
        print(f"\n  [{i}] {pocket_fn}")
        print(f"       {ligand_fn}")

        if args.model == 'diffsbdd':
            lig_data, poc_data = process_ligand_and_pocket(
                pdb_path, sdf_path, atom_dict=atom_dict,
                dist_cutoff=args.dist_cutoff, ca_only=False)
            pc, ph = poc_data['pocket_coords'], poc_data['pocket_one_hot']
            lc, lh = lig_data['lig_coords'],    lig_data['lig_one_hot']
        elif args.model == 'targetdiff':
            pc, ph = featurize_pocket_targetdiff(pdb_path)
            lc, lh = featurize_ligand(sdf_path, atom_dict)

        print(f"      poc_coords  {pc.shape}  poc_one_hot {ph.shape}")
        print(f"      lig_coords  {lc.shape}  lig_one_hot {lh.shape}")

        assert pc.ndim == 2 and pc.shape[1] == 3, f"poc_coords must be [N,3], got {pc.shape}"
        assert ph.ndim == 2, f"poc_one_hot must be 2-D, got {ph.shape}"
        assert lc.ndim == 2 and lc.shape[1] == 3, f"lig_coords must be [M,3], got {lc.shape}"
        assert pc.shape[0] == ph.shape[0], "poc_coords / poc_one_hot atom-count mismatch"

        poc_list.append(pc); lig_list.append(lc)

    print("\nVerifying concatenation...")
    np.concatenate(poc_list); np.concatenate(lig_list)
    print("Smoke test PASSED — shapes correct, concatenation works.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build an NPZ dataset from CrossDocked pocket10.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--crossdocked_dir', required=True,
                        help='Path to crossdocked_pocket10/ directory')
    parser.add_argument('--split_path', required=True,
                        help='Path to split_by_name.pt')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--model', choices=['diffsbdd', 'targetdiff'], default='diffsbdd',
                        help='Pocket featurisation format')
    parser.add_argument('--dist_cutoff', type=float, default=8.0,
                        help='Pocket residue distance cutoff in Å (DiffSBDD path only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for val-split sampling')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run on 2 pairs only to verify shapes')
    args = parser.parse_args()

    random.seed(args.seed)
    if args.smoke_test:
        smoke_test(args)
    else:
        main(args)
