"""
test.py — Generate ligands for the CrossDocked test set.

Scans a test directory for (*.sdf, *.pdb, *.txt) triplets and generates
n_samples molecules per pocket. Mirrors test_targets.py but discovers pockets
from a directory scan rather than a fixed target list.

Expected layout under --test_dir:
    <test_dir>/
    ├── <pdb_id>_<lig>_<chain>_<resid>.sdf   # reference ligand
    ├── <pdb_id>.pdb                           # pocket structure
    └── <pdb_id>_<lig>_<chain>_<resid>.txt    # pocket residue list

Usage
-----
    python -m scripts.test \\
        checkpoints/my_run.ckpt \\
        --config configs/ppo_config.yaml \\
        --test_dir /data/crossdock/test \\
        --outdir results/crossdock_test \\
        --n_samples 100 \\
        --batch_size 120 \\
        --sanitize
"""

import argparse
import warnings
import sys
from pathlib import Path
from time import time

import torch
import numpy as np
import yaml
from tqdm import tqdm
from openbabel import openbabel

openbabel.obErrorLog.StopLogging()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DIFFSBDD_ROOT = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_DIFFSBDD_ROOT))

from src.prism.utils import dict_to_namespace, write_sdf_file
from src.prism.models.policy_factory import build_diffsbdd_policy
from src.models.diffsbdd.analysis.molecule_builder import process_molecule

MAXITER = 10
MAXNTRIES = 10


def load_diffsbdd(checkpoint_path: Path, config_path: Path, device: str):
    """Build DiffSBDD via the shared factory and return the raw ddpm_module."""
    with open(config_path) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    histogram_path = Path(config.datadir) / "size_distribution.npy"
    if not histogram_path.exists():
        raise FileNotFoundError(f"Size distribution not found: {histogram_path}")
    node_histogram = np.load(histogram_path).tolist()

    _, ddpm_module, _ = build_diffsbdd_policy(
        config=config,
        device=torch.device(device),
        node_histogram=node_histogram,
        warm_start_checkpoint=str(checkpoint_path),
    )
    ddpm_module.eval()
    return ddpm_module


def generate_for_pocket(
    ddpm_module,
    sdf_file: Path,
    pdb_file: Path,
    resi_list: list,
    outdir: Path,
    n_samples: int,
    batch_size: int,
    sanitize: bool,
    relax: bool,
    all_frags: bool,
    fix_n_nodes: bool,
    timesteps: int | None,
    resamplings: int,
    jump_length: int,
    n_nodes_bias: int,
    n_nodes_min: int,
    skip_existing: bool,
) -> float | None:
    """
    Generate n_samples valid molecules for one CrossDocked pocket.

    Returns elapsed time in seconds, or None if skipped.
    Retries up to MAXNTRIES times on failure.
    """
    ligand_name = sdf_file.stem
    raw_dir       = outdir / "raw"
    processed_dir = outdir / "processed"
    times_dir     = outdir / "pocket_times"

    raw_out       = raw_dir       / f"{ligand_name}_gen.sdf"
    processed_out = processed_dir / f"{ligand_name}_gen.sdf"
    time_file     = times_dir     / f"{ligand_name}.txt"

    if skip_existing and time_file.exists() and processed_out.exists():
        with open(time_file) as f:
            return float(f.read().split()[1])

    if fix_n_nodes:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        num_nodes_lig = suppl[0].GetNumAtoms() if suppl[0] else None
    else:
        num_nodes_lig = None

    for attempt in range(MAXNTRIES):
        try:
            t_start = time()

            all_molecules: list  = []
            valid_molecules: list = []
            processed_molecules: list = []
            n_generated = 0
            n_valid = 0
            iter_count = 0

            while len(valid_molecules) < n_samples:
                iter_count += 1
                if iter_count > MAXITER:
                    raise RuntimeError("Maximum iterations exceeded.")

                num_nodes_batch = (
                    torch.ones(batch_size, dtype=torch.int) * num_nodes_lig
                    if num_nodes_lig is not None
                    else None
                )

                with torch.no_grad():
                    mols_batch = ddpm_module.generate_ligands(
                        pdb_file,
                        batch_size,
                        resi_list,
                        num_nodes_lig=num_nodes_batch,
                        timesteps=timesteps,
                        sanitize=False,
                        largest_frag=False,
                        relax_iter=0,
                        n_nodes_bias=n_nodes_bias,
                        n_nodes_min=n_nodes_min,
                        resamplings=resamplings,
                        jump_length=jump_length,
                    )

                all_molecules.extend(mols_batch)

                batch_processed = [
                    process_molecule(
                        m,
                        sanitize=sanitize,
                        relax_iter=(200 if relax else 0),
                        largest_frag=not all_frags,
                    )
                    for m in mols_batch
                ]
                processed_molecules.extend(batch_processed)

                valid_batch = [m for m in batch_processed if m is not None]
                n_generated += batch_size
                n_valid += len(valid_batch)
                valid_molecules.extend(valid_batch)

            valid_molecules = valid_molecules[:n_samples]

            # Valid molecules first in the raw SDF (easier for downstream scoring)
            all_molecules = (
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is not None] +
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is None]
            )

            write_sdf_file(raw_out, all_molecules)
            write_sdf_file(processed_out, valid_molecules)

            elapsed = time() - t_start
            with open(time_file, "w") as f:
                f.write(f"{sdf_file} {elapsed}")
            return elapsed

        except (RuntimeError, ValueError) as e:
            if attempt >= MAXNTRIES - 1:
                raise RuntimeError(f"Max retries exceeded for {ligand_name}: {e}")
            warnings.warn(f"Attempt {attempt+1}/{MAXNTRIES} failed: {e}. Retrying…")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands for the CrossDocked test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path,
                        help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--config", type=Path, required=True,
                        help="PRISM YAML config")
    parser.add_argument("--test_dir", type=Path, required=True,
                        help="CrossDocked test directory (contains .sdf, .pdb, .txt files)")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, default=None,
                        help="Optional file of ligand stems to restrict generation")
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=120)
    parser.add_argument("--sanitize",    action="store_true")
    parser.add_argument("--relax",       action="store_true")
    parser.add_argument("--all_frags",   action="store_true")
    parser.add_argument("--fix_n_nodes", action="store_true")
    parser.add_argument("--timesteps",   type=int, default=None)
    parser.add_argument("--resamplings", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--n_nodes_bias", type=int, default=0)
    parser.add_argument("--n_nodes_min",  type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    (args.outdir / "raw").mkdir(parents=True, exist_ok=True)
    (args.outdir / "processed").mkdir(exist_ok=True)
    (args.outdir / "pocket_times").mkdir(exist_ok=True)

    ddpm_module = load_diffsbdd(args.checkpoint, args.config, device)

    test_files = sorted(args.test_dir.glob("[!.]*.sdf"))
    if args.test_list is not None:
        with open(args.test_list) as f:
            allowed = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in allowed]

    print(f"\nFound {len(test_files)} pockets.")

    time_per_pocket: dict = {}
    pbar = tqdm(test_files, desc="Pockets")

    for sdf_file in pbar:
        ligand_name = sdf_file.stem
        pdb_id      = ligand_name.split("_")[0]
        pdb_file    = sdf_file.parent / f"{pdb_id}.pdb"
        txt_file    = sdf_file.parent / f"{ligand_name}.txt"

        if not pdb_file.exists():
            warnings.warn(f"[SKIP] PDB not found: {pdb_file}")
            continue
        if not txt_file.exists():
            warnings.warn(f"[SKIP] Residue list not found: {txt_file}")
            continue

        with open(txt_file) as f:
            resi_list = f.read().split()

        try:
            elapsed = generate_for_pocket(
                ddpm_module=ddpm_module,
                sdf_file=sdf_file,
                pdb_file=pdb_file,
                resi_list=resi_list,
                outdir=args.outdir,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                sanitize=args.sanitize,
                relax=args.relax,
                all_frags=args.all_frags,
                fix_n_nodes=args.fix_n_nodes,
                timesteps=args.timesteps,
                resamplings=args.resamplings,
                jump_length=args.jump_length,
                n_nodes_bias=args.n_nodes_bias,
                n_nodes_min=args.n_nodes_min,
                skip_existing=args.skip_existing,
            )
            if elapsed is not None:
                time_per_pocket[str(sdf_file)] = elapsed
                pbar.set_description(f"Last: {ligand_name} ({elapsed:.1f}s)")
        except Exception as e:
            warnings.warn(f"[ERROR] {ligand_name}: {e}")

    times_summary = args.outdir / "pocket_times.txt"
    with open(times_summary, "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    if time_per_pocket:
        times = list(time_per_pocket.values())
        mean_t = np.mean(times)
        std_t  = np.std(times)
        print(f"\nTime per pocket: {mean_t:.1f} ± {std_t:.1f}s")
        print(f"Completed: {len(time_per_pocket)} pockets")
        print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
