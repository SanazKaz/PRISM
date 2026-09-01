"""
generate_targetdiff.py — Standalone ligand generation with a TargetDiff checkpoint.

Usage
-----
    python -m scripts.generate_targetdiff \\
        checkpoints/targetdiff.pt \\
        --config configs/targetdiff_ppo.yaml \\
        --pdbfile data/my_dataset/02_preprocessed/pocket_files/1cil_ETS_C_263_pocket.pdb \\
        --outfile results/generated.sdf \\
        --n_samples 100 \\
        --batch_size 25
"""

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT    = Path(__file__).resolve().parents[1]
_TARGETDIFF_ROOT = _PROJECT_ROOT / "src" / "models" / "targetdiff"
_DIFFSBDD_ROOT   = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_TARGETDIFF_ROOT))
sys.path.insert(2, str(_DIFFSBDD_ROOT))

from src.prism.utils import write_sdf_file
from src.prism.models.targetdiff_inference import (
    load_targetdiff_model, pocket_from_pdb, reconstruct_molecules,
)

from utils import transforms as trans                          # noqa: E402
from scripts.sample_diffusion import sample_diffusion_ligand   # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands from a TargetDiff checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path,
                        help="Path to TargetDiff checkpoint (.pt)")
    parser.add_argument("--config", type=Path, required=True,
                        help="PRISM TargetDiff YAML config (e.g. configs/targetdiff_ppo.yaml)")
    parser.add_argument("--pdbfile", type=str, required=True,
                        help="Protein or pocket PDB file (ligand-free)")
    parser.add_argument("--ref_ligand", type=str, default=None,
                        help="Reference ligand SDF. The pocket is cut to residues "
                             "within 10 A of it, matching TargetDiff training. "
                             "Strongly recommended: omitting it on a full protein "
                             "or a differently-cut pocket sharply lowers validity.")
    parser.add_argument("--outfile", type=Path, required=True,
                        help="Output SDF file path")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Diffusion steps (default: 1000, matches TargetDiff training)")
    parser.add_argument("--sample_num_atoms", type=str, default="prior",
                        choices=["prior"],
                        help="How to pick ligand size: 'prior' samples from the "
                             "learned size distribution")
    parser.add_argument("--center_pos_mode", type=str, default="protein",
                        help="How to centre coordinates before sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_targetdiff_model(args.checkpoint, args.config, device)
    protein_featurizer = trans.FeaturizeProteinAtom()

    if args.ref_ligand is None:
        print("WARNING: no --ref_ligand given; the PDB is used as-is. "
              "Pass one unless it is already a TargetDiff-style cut pocket.")
    print(f"Parsing pocket: {args.pdbfile}")
    pocket_data = pocket_from_pdb(args.pdbfile, protein_featurizer,
                                  ref_ligand_sdf=args.ref_ligand)

    print(f"Generating {args.n_samples} molecules in batches of {args.batch_size}…")
    with torch.no_grad():
        all_pred_pos, all_pred_v, *_ = sample_diffusion_ligand(
            model=model,
            data=pocket_data,
            num_samples=args.n_samples,
            batch_size=args.batch_size,
            device=device,
            num_steps=args.num_steps,
            pos_only=False,
            center_pos_mode=args.center_pos_mode,
            sample_num_atoms=args.sample_num_atoms,
        )

    molecules = reconstruct_molecules(all_pred_pos, all_pred_v)
    valid = [m for m in molecules if m is not None]
    print(f"Reconstruction: {len(valid)}/{len(molecules)} valid molecules")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    write_sdf_file(args.outfile, valid)
    print(f"Saved {len(valid)} molecules to {args.outfile}")


if __name__ == "__main__":
    main()
