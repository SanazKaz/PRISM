"""
generate_diffsbdd.py — Standalone ligand generation with a DiffSBDD checkpoint.

Usage
-----
From a PRISM fine-tuned checkpoint:

    python -m scripts.generate_diffsbdd \\
        checkpoints/my_run.ckpt \\
        --config configs/ppo_config.yaml \\
        --pdbfile data/my_dataset/02_preprocessed/pocket_files/1cil_ETS_C_263_pocket.pdb \\
        --ref_ligand data/my_dataset/02_preprocessed/sdf_files/1cil_ETS_C_263.sdf \\
        --outfile results/generated.sdf \\
        --n_samples 100 \\
        --batch_size 25 \\
        --sanitize

From the original DiffSBDD pretrained checkpoint:

    python -m scripts.generate_diffsbdd \\
        checkpoints/crossdocked_fullatom_cond.ckpt \\
        --config configs/ppo_config.yaml \\
        --pdbfile path/to/pocket.pdb \\
        --outfile results/generated.sdf
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np
import yaml
from rdkit import Chem
from tqdm import tqdm

# Add project root and DiffSBDD to sys.path so vendored DiffSBDD imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DIFFSBDD_ROOT = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_DIFFSBDD_ROOT))

from src.prism.utils import dict_to_namespace, write_sdf_file
from src.prism.models.policy_factory import build_diffsbdd_policy
from src.models.diffsbdd.analysis.molecule_builder import process_molecule


def load_diffsbdd(checkpoint_path: Path, config_path: Path, device: str):
    """
    Build a DiffSBDD policy from a config + checkpoint using the shared factory.

    This is the single correct way to load DiffSBDD in PRISM. The factory:
      - Injects architecture defaults so configs stay minimal
      - Converts egnn_params / diffusion_params dicts to Namespaces (required
        by LigandPocketDDPM's attribute-style access)
      - Supplies lr from config.ppo.lr so LigandPocketDDPM's positional arg is met
      - Handles all checkpoint prefix variants (.pt, .ckpt, ddpm_model., policy._ddpm_module.)

    Returns the raw ddpm_module (LigandPocketDDPM), which exposes generate_ligands().
    """
    with open(config_path) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    histogram_path = Path(config.datadir) / "size_distribution.npy"
    if not histogram_path.exists():
        raise FileNotFoundError(
            f"Size distribution histogram not found at {histogram_path}. "
            "Run process_data.py first, or set config.datadir correctly."
        )
    node_histogram = np.load(histogram_path).tolist()

    device_obj = torch.device(device)
    # build_diffsbdd_policy returns (policy, ddpm_module, dataset_info).
    # For generation we only need ddpm_module — it exposes generate_ligands()
    # which handles PDB parsing, pocket featurisation, diffusion, and atom decoding.
    _, ddpm_module, _ = build_diffsbdd_policy(
        config=config,
        device=device_obj,
        node_histogram=node_histogram,
        warm_start_checkpoint=str(checkpoint_path),
    )
    ddpm_module.eval()
    return ddpm_module


def generate(
    ddpm_module,
    pdbfile: str,
    ref_ligand: str | None,
    n_samples: int,
    batch_size: int,
    num_nodes_lig: int | None,
    timesteps: int | None,
    sanitize: bool,
    relax: bool,
    all_frags: bool,
    resamplings: int,
    jump_length: int,
    save_traj: bool,
    resi_list: list | None,
) -> list:
    """
    Run the DiffSBDD reverse diffusion loop and return a list of RDKit molecules.

    We generate in batches and stop once we have n_samples. The batch loop is
    needed because some generations produce invalid molecules and are discarded.
    """
    # Build a fixed-size tensor for the requested atom count if provided.
    # None lets DiffSBDD sample from its learned size distribution.
    nodes_tensor = (
        torch.ones(n_samples, dtype=torch.int) * num_nodes_lig
        if num_nodes_lig is not None
        else None
    )

    molecules: list[Chem.Mol] = []
    n_batches = int(np.ceil(n_samples / batch_size))

    with tqdm(total=n_samples, desc="Generating") as pbar:
        for i in range(n_batches):
            current_bs = min(batch_size, n_samples - len(molecules))
            batch_nodes = (
                nodes_tensor[len(molecules): len(molecules) + current_bs]
                if nodes_tensor is not None
                else None
            )

            with torch.no_grad():
                batch = ddpm_module.generate_ligands(
                    pdbfile,
                    current_bs,
                    resi_list,
                    ref_ligand,
                    batch_nodes,
                    sanitize,
                    largest_frag=not all_frags,
                    relax_iter=(200 if relax else 0),
                    resamplings=resamplings,
                    jump_length=jump_length,
                    timesteps=timesteps,
                    save_traj=save_traj,
                )
            molecules.extend(batch)
            pbar.update(len(batch))

    return molecules


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands from a DiffSBDD (PRISM or upstream) checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pt or .ckpt checkpoint")
    parser.add_argument("--config", type=Path, required=True,
                        help="PRISM YAML config (e.g. configs/ppo_config.yaml)")
    parser.add_argument("--pdbfile", type=str, required=True,
                        help="Pocket PDB file (ligand-free)")
    parser.add_argument("--outfile", type=Path, required=True,
                        help="Output SDF file path")
    parser.add_argument("--ref_ligand", type=str, default=None,
                        help="Reference ligand SDF (used to orient the pocket)")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--num_nodes_lig", type=int, default=None,
                        help="Fixed atom count; defaults to DiffSBDD's size distribution")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Diffusion steps; defaults to model's total_timesteps")
    parser.add_argument("--resi_list", nargs="+", default=None,
                        help="Residue identifiers to define the pocket (optional)")
    parser.add_argument("--sanitize", action="store_true",
                        help="Sanitize generated molecules with RDKit")
    parser.add_argument("--relax", action="store_true",
                        help="Run UFF force-field relaxation (200 iterations)")
    parser.add_argument("--all_frags", action="store_true",
                        help="Keep all fragments (default: largest only)")
    parser.add_argument("--resamplings", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--save_traj", action="store_true",
                        help="Save the full diffusion trajectory (forces n_samples=1)")
    args = parser.parse_args()

    if args.save_traj and args.n_samples > 1:
        print("[INFO] --save_traj forces n_samples=1")
        args.n_samples = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ddpm_module = load_diffsbdd(args.checkpoint, args.config, device)

    molecules = generate(
        ddpm_module=ddpm_module,
        pdbfile=args.pdbfile,
        ref_ligand=args.ref_ligand,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_nodes_lig=args.num_nodes_lig,
        timesteps=args.timesteps,
        sanitize=args.sanitize,
        relax=args.relax,
        all_frags=args.all_frags,
        resamplings=args.resamplings,
        jump_length=args.jump_length,
        save_traj=args.save_traj,
        resi_list=args.resi_list,
    )

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    write_sdf_file(args.outfile, molecules)
    print(f"Saved {len(molecules)} molecules to {args.outfile}")


if __name__ == "__main__":
    main()
