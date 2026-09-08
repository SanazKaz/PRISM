"""
Reference 2D property statistics for the Property2DReward.

Computes per-target mean/std of the normalised molecular properties used by
`property_match.Property2DReward` and writes them to a single JSON keyed by
target name.

Adapted from
https://github.com/jianingli-purdue/Benchmarking_gene_model/blob/main/Picture_drawing.ipynb
Yang et al. https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c01706

Every property is divided by its denominator below BEFORE the statistics are
taken, so the stored means are on the same scale as the values the reward
computes at training time. Keep this dict in sync with
`Property2DReward._calculate_properties`.

Usage:
    python -m src.prism.reward.scoring.reward_data.property_ref_creator \
        --ref_dir data --output_json \
        src/prism/reward/scoring/reward_data/propeties_ref.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402  - same SA implementation the reward uses

RDLogger.DisableLog('rdApp.*')


# ============== PROPERTY DEFINITIONS ==============
PROPERTY_DENOMINATORS = {
    'MW': 500, 'AliR_C': 4, 'AroR_C': 3, 'ChiA_C': 6, 'SA': 6,
    'NHOH_C': 6, 'HetA_C': 10, 'RotB_C': 8, 'BriA_C': 2,
    'HBD_C': 5, 'HBA_C': 10, 'FusedR_C': 6, 'LogP_C': 5,
}


def compute_properties(mol: Chem.Mol) -> dict:
    """Normalised 2D properties for one molecule, or None if it cannot be read."""
    if mol is None:
        return None
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        ring_info = mol.GetRingInfo()

        raw = {
            'MW':       Descriptors.MolWt(mol),
            'AliR_C':   Lipinski.NumAliphaticRings(mol),
            'AroR_C':   Lipinski.NumAromaticRings(mol),
            'ChiA_C':   len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'SA':       sascorer.calculateScore(mol),
            'NHOH_C':   Lipinski.NHOHCount(mol),
            'HetA_C':   sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
            'RotB_C':   Descriptors.NumRotatableBonds(mol),
            'BriA_C':   rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            'HBD_C':    rdMolDescriptors.CalcNumHBD(mol),
            'HBA_C':    rdMolDescriptors.CalcNumHBA(mol),
            'FusedR_C': sum(1 for i in range(ring_info.NumRings())
                            if ring_info.IsRingFused(i)),
            'LogP_C':   Crippen.MolLogP(mol),
        }
        return {k: v / PROPERTY_DENOMINATORS[k] for k, v in raw.items()}
    except Exception as e:
        print(f"  Warning: could not compute properties: {e}")
        return None


def compute_statistics(molecules: list) -> dict:
    """Per-property statistics over a set of molecules."""
    values = {p: [] for p in PROPERTY_DENOMINATORS}
    for mol in molecules:
        props = compute_properties(mol)
        if props is not None:
            for k, v in props.items():
                values[k].append(v)

    stats = {}
    for prop, v in values.items():
        v = np.asarray(v, dtype=float)
        stats[prop] = {
            'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'min': float(np.min(v)), 'max': float(np.max(v)),
            'range': float(np.max(v) - np.min(v)),
            'median': float(np.median(v)), 'n_samples': int(v.size),
        } if v.size else dict.fromkeys(
            ('mean', 'std', 'min', 'max', 'range', 'median'), 0.0) | {'n_samples': 0}
    return stats


# ============== IO ==============
def load_target_molecules(ref_dir: Path, sdf_subdir: str) -> dict:
    """Map target name -> molecules.

    A target is any <ref_dir>/<TARGET>/<sdf_subdir> directory. Subdirectories
    without one are skipped, so unrelated dataset directories are ignored.
    """
    targets = {}
    for entry in sorted(d for d in ref_dir.iterdir() if d.is_dir()):
        sdf_dir = entry / sdf_subdir
        if not sdf_dir.is_dir():
            print(f"  skipping {entry.name}: no {sdf_subdir}/")
            continue
        sdfs = sorted(sdf_dir.glob('*.sdf'))
        if not sdfs:
            continue
        mols = [m for f in sdfs for m in Chem.SDMolSupplier(str(f)) if m is not None]
        if mols:
            targets[entry.name] = mols
            print(f"  {entry.name}: {len(mols)} molecules from {len(sdfs)} file(s)")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref_dir', required=True,
                        help='Directory of reference ligands, one SDF or subdirectory per target')
    parser.add_argument('--output_json', required=True, help='Output JSON path')
    parser.add_argument('--sdf_subdir', default='02_preprocessed/sdf_files',
                        help='Per-target subdirectory holding the reference SDFs')
    args = parser.parse_args()

    print(f"Loading reference ligands from {args.ref_dir}")
    targets = load_target_molecules(Path(args.ref_dir), args.sdf_subdir)
    if not targets:
        raise FileNotFoundError(f"No SDF files found under {args.ref_dir}")

    all_stats = {name: compute_statistics(mols) for name, mols in targets.items()}

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nWrote {len(all_stats)} targets to {out}")


if __name__ == '__main__':
    main()
