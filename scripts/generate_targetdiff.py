"""
generate_targetdiff.py — Standalone ligand generation with a TargetDiff checkpoint.

TargetDiff uses a different pocket featuriser than DiffSBDD (27-dim protein atom
features vs DiffSBDD's 10-dim element one-hots), so generation cannot go through
the shared BaseDiffusionPolicy.sample_given_pocket interface — that interface
operates on pre-processed NPZ tensors used during PPO training. For stand-alone
inference from a raw PDB file, we use TargetDiff's own PDBProtein parser and
sample_diffusion_ligand pipeline instead.

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
import yaml
from rdkit import Chem
from tqdm import tqdm

# Add project root and TargetDiff source to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TARGETDIFF_ROOT = _PROJECT_ROOT / "src" / "models" / "targetdiff"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_TARGETDIFF_ROOT))

from src.prism.utils import dict_to_namespace, write_sdf_file
from src.prism.models.policy_factory import build_targetdiff_policy

# TargetDiff internal imports — available after sys.path.insert above.
from utils.data import PDBProtein                          # noqa: E402
from datasets.pl_data import ProteinLigandData, torchify_dict  # noqa: E402
from utils import transforms as trans                      # noqa: E402
from scripts.sample_diffusion import sample_diffusion_ligand  # noqa: E402
from utils import reconstruct                              # noqa: E402


def _pocket_from_pdb(pdb_path: str, protein_featurizer) -> "ProteinLigandData":
    """
    Parse a pocket PDB file into the ProteinLigandData format that
    sample_diffusion_ligand expects.

    The protein featurizer converts raw atom dicts into the 27-dim feature
    vectors that TargetDiff's ScorePosNet3D was trained on. An empty ligand
    placeholder is required by TargetDiff's data class (the model generates
    the ligand from scratch, so it just needs zeros).
    """
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            "element":      torch.empty([0], dtype=torch.long),
            "pos":          torch.empty([0, 3], dtype=torch.float),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
            "bond_index":   torch.empty([2, 0], dtype=torch.long),
            "bond_type":    torch.empty([0], dtype=torch.long),
        },
    )
    data = protein_featurizer(data)
    return data


def _reconstruct_molecules(all_pred_pos, all_pred_v) -> list:
    """
    Convert raw TargetDiff position + atom-type predictions into RDKit molecules.

    TargetDiff encodes atom types as indices into MAP_ATOM_TYPE_AROMATIC_TO_INDEX
    (13 classes). get_atomic_number_from_index decodes these back to element
    numbers, and is_aromatic_from_index recovers the aromaticity flag that
    reconstruct_from_generated uses to set bond orders correctly.
    """
    molecules = []
    for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
        try:
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            pred_aromatic  = trans.is_aromatic_from_index(pred_v, mode="add_aromatic")
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            # Discard disconnected molecules (salts / fragments)
            if "." not in smiles:
                molecules.append(mol)
            else:
                molecules.append(None)
        except reconstruct.MolReconsError:
            molecules.append(None)
    return molecules


def load_targetdiff(checkpoint_path: Path, config_path: Path, device: str):
    """
    Build a TargetDiff policy using the shared factory and return the inner
    ScorePosNet3D model ready for inference.

    build_targetdiff_policy resolves the checkpoint path from config.model.checkpoint
    (or falls back to the CLI checkpoint argument), loads weights with the
    correct strict/non-strict logic, and returns a TargetDiffPolicy wrapper.
    We then reach into policy._model to get the raw ScorePosNet3D needed by
    sample_diffusion_ligand.
    """
    with open(config_path) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    # Pass the CLI checkpoint as the warm_start fallback in case config.model.checkpoint
    # is not set (e.g. when running against the original upstream checkpoint).
    policy, _ = build_targetdiff_policy(
        config=config,
        device=torch.device(device),
        warm_start_checkpoint=str(checkpoint_path),
    )
    policy.eval()
    return policy._model


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
                        help="Pocket PDB file (ligand-free)")
    parser.add_argument("--outfile", type=Path, required=True,
                        help="Output SDF file path")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Diffusion steps (default: 1000, matches TargetDiff training)")
    parser.add_argument("--sample_num_atoms", type=str, default="prior",
                        choices=["prior", "ref"],
                        help="How to pick ligand size: 'prior' samples from learned "
                             "distribution; 'ref' requires a reference ligand")
    parser.add_argument("--center_pos_mode", type=str, default="protein",
                        help="How to centre coordinates before sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_targetdiff(args.checkpoint, args.config, device)

    # TargetDiff's protein featurizer produces the 27-dim vectors the model expects.
    protein_featurizer = trans.FeaturizeProteinAtom()

    print(f"Parsing pocket: {args.pdbfile}")
    pocket_data = _pocket_from_pdb(args.pdbfile, protein_featurizer)

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

    molecules = _reconstruct_molecules(all_pred_pos, all_pred_v)
    valid = [m for m in molecules if m is not None]
    print(f"Reconstruction: {len(valid)}/{len(molecules)} valid molecules")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    write_sdf_file(args.outfile, valid)
    print(f"Saved {len(valid)} molecules to {args.outfile}")


if __name__ == "__main__":
    main()
