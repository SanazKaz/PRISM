"""
test_prism_crossdocked.py

Generate ligands for the CrossDocked test set using a PRISM .pt checkpoint.
Mirrors the structure of test_prism_targets.py but loops over the crossdocked
test directory (PDB + SDF + TXT files) rather than hardcoded targets.
"""

import argparse
import warnings
import inspect
import yaml
import sys
import os
from pathlib import Path
from time import time
from argparse import Namespace

import torch
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from openbabel import openbabel

# Suppress OpenBabel logging
openbabel.obErrorLog.StopLogging()

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFSBDD_PATH = Path(PROJECT_ROOT) / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(DIFFSBDD_PATH))

# Patch missing visualization function before importing lightning_modules
import analysis.visualization as vis_module
if not hasattr(vis_module, 'visualize_ligand_only'):
    vis_module.visualize_ligand_only = lambda *args, **kwargs: None

# --- Imports ---
try:
    from lightning_modules import LigandPocketDDPM
    from analysis.molecule_builder import process_molecule
    import utils
except ImportError:
    from src.models.diffsbdd.lightning_modules import LigandPocketDDPM
    from src.models.diffsbdd.analysis.molecule_builder import process_molecule
    from src.models.diffsbdd import utils

MAXITER = 10
MAXNTRIES = 10


class RecursiveNamespace(Namespace):
    """Recursively converts a dictionary to a Namespace for nested config access."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                setattr(self, k, RecursiveNamespace(**v))
            elif isinstance(v, list):
                setattr(self, k, [RecursiveNamespace(**i) if isinstance(i, dict) else i for i in v])


def load_model(checkpoint_path: Path, config_path: Path, device: str):
    """
    Initialize model from config and load weights from .pt checkpoint.

    Args:
        checkpoint_path: Path to model weights (.pt or .ckpt)
        config_path: Path to model configuration YAML
        device: Device to load model onto ('cuda' or 'cpu')

    Returns:
        Loaded and initialised LigandPocketDDPM model
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    full_config_ns = RecursiveNamespace(**raw_config)

    sig = inspect.signature(LigandPocketDDPM.__init__)
    valid_keys = set(sig.parameters.keys()) - {'self'}

    if hasattr(full_config_ns, 'model') and isinstance(full_config_ns.model, Namespace):
        source_params = full_config_ns.model.__dict__
    else:
        source_params = full_config_ns.__dict__

    filtered_args = {k: v for k, v in source_params.items() if k in valid_keys}

    if 'datadir' in filtered_args:
        datadir = Path(filtered_args['datadir'])
        histogram_path = datadir / 'size_distribution.npy'
        if histogram_path.exists():
            filtered_args['node_histogram'] = np.load(histogram_path).tolist()
        else:
            print(f"[WARNING] Size distribution not found at {histogram_path}")

    try:
        model = LigandPocketDDPM(**filtered_args)
    except TypeError as e:
        raise RuntimeError(f"Model initialisation failed: {e}")

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("[INFO] Detected Lightning checkpoint format.")
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ddpm_model.'):
                new_state_dict[k.replace('ddpm_model.', '')] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    else:
        print("[INFO] Detected raw state_dict format.")
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
        print("[SUCCESS] Weights loaded successfully.")
    except RuntimeError as e:
        print(f"[INFO] Strict load failed: {e}")
        print("[INFO] Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def generate_for_pocket(
    model,
    sdf_file: Path,
    pdb_file: Path,
    resi_list: list,
    outdir: Path,
    n_samples: int = 100,
    batch_size: int = 100,
    sanitize: bool = True,
    relax: bool = False,
    all_frags: bool = False,
    fix_n_nodes: bool = False,
    timesteps: int = None,
    resamplings: int = 10,
    jump_length: int = 1,
    n_nodes_bias: int = 0,
    n_nodes_min: int = 0,
    skip_existing: bool = False,
):
    """
    Generate molecules for a single crossdocked pocket.

    Args:
        model: Loaded LigandPocketDDPM model
        sdf_file: Path to reference ligand SDF
        pdb_file: Path to pocket PDB file
        resi_list: List of pocket residue identifiers from .txt file
        outdir: Output directory for raw and processed SDFs
        n_samples: Number of valid molecules to generate
        batch_size: Generation batch size
        sanitize: Whether to sanitize molecules with RDKit
        relax: Whether to run force field relaxation
        all_frags: Whether to keep all fragments
        fix_n_nodes: Whether to fix atom count to reference ligand
        timesteps: Number of diffusion timesteps
        resamplings: Number of resamplings per step
        jump_length: Jump length for resampling
        n_nodes_bias: Bias added to sampled node count
        n_nodes_min: Minimum node count
        skip_existing: Skip if output already exists

    Returns:
        Time taken for this pocket, or None if skipped
    """
    ligand_name = sdf_file.stem
    raw_sdf_dir = outdir / 'raw'
    processed_sdf_dir = outdir / 'processed'
    times_dir = outdir / 'pocket_times'

    sdf_out_raw = raw_sdf_dir / f'{ligand_name}_gen.sdf'
    sdf_out_processed = processed_sdf_dir / f'{ligand_name}_gen.sdf'
    time_file = times_dir / f'{ligand_name}.txt'

    if skip_existing and time_file.exists() and sdf_out_processed.exists() and sdf_out_raw.exists():
        with open(time_file, 'r') as f:
            elapsed = float(f.read().split()[1])
        return elapsed

    if fix_n_nodes:
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        num_nodes_lig = suppl[0].GetNumAtoms() if suppl[0] else None
    else:
        num_nodes_lig = None

    for n_try in range(MAXNTRIES):
        try:
            t_start = time()

            all_molecules = []
            valid_molecules = []
            processed_molecules = []
            iter_count = 0
            n_generated = 0
            n_valid = 0

            while len(valid_molecules) < n_samples:
                iter_count += 1
                if iter_count > MAXITER:
                    raise RuntimeError('Maximum number of iterations exceeded.')

                num_nodes_lig_inflated = None if num_nodes_lig is None else \
                    torch.ones(batch_size, dtype=int) * num_nodes_lig

                with torch.no_grad():
                    mols_batch = model.generate_ligands(
                        pdb_file,
                        batch_size,
                        resi_list,
                        num_nodes_lig=num_nodes_lig_inflated,
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

                mols_batch_processed = [
                    process_molecule(
                        m,
                        sanitize=sanitize,
                        relax_iter=(200 if relax else 0),
                        largest_frag=not all_frags,
                    )
                    for m in mols_batch
                ]
                processed_molecules.extend(mols_batch_processed)

                valid_batch = [m for m in mols_batch_processed if m is not None]
                n_generated += batch_size
                n_valid += len(valid_batch)
                valid_molecules.extend(valid_batch)

            # Trim to exact count
            valid_molecules = valid_molecules[:n_samples]

            # Reorder: valid first, then invalid
            all_molecules = \
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is not None] + \
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is None]

            utils.write_sdf_file(sdf_out_raw, all_molecules)
            utils.write_sdf_file(sdf_out_processed, valid_molecules)

            elapsed = time() - t_start
            with open(time_file, 'w') as f:
                f.write(f"{str(sdf_file)} {elapsed}")

            return elapsed

        except (RuntimeError, ValueError) as e:
            if n_try >= MAXNTRIES - 1:
                raise RuntimeError(f"Max retries exceeded for {ligand_name}: {e}")
            warnings.warn(f"Attempt {n_try + 1}/{MAXNTRIES} failed: '{e}'. Retrying...")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands for the CrossDocked test set using a PRISM .pt checkpoint"
    )
    parser.add_argument('checkpoint', type=Path,
                        help="Path to model weights (.pt or .ckpt)")
    parser.add_argument('--config', type=Path, required=True,
                        help="Path to model config YAML (e.g. ppo_config.yaml)")
    parser.add_argument('--test_dir', type=Path, required=True,
                        help="Path to crossdocked test directory containing PDB/SDF/TXT files")
    parser.add_argument('--outdir', type=Path, required=True,
                        help="Output directory")
    parser.add_argument('--test_list', type=Path, default=None,
                        help="Optional comma-separated list of ligand stems to restrict generation")
    parser.add_argument('--n_samples', type=int, default=100,
                        help="Number of valid molecules per pocket (default: 100)")
    parser.add_argument('--batch_size', type=int, default=120,
                        help="Batch size for generation (default: 120)")
    parser.add_argument('--sanitize', action='store_true',
                        help="Sanitize molecules with RDKit")
    parser.add_argument('--relax', action='store_true',
                        help="Run force field relaxation")
    parser.add_argument('--all_frags', action='store_true',
                        help="Keep all fragments")
    parser.add_argument('--fix_n_nodes', action='store_true',
                        help="Fix atom count to reference ligand")
    parser.add_argument('--timesteps', type=int, default=None,
                        help="Diffusion timesteps")
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--n_nodes_bias', type=int, default=0)
    parser.add_argument('--n_nodes_min', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip pockets that already have output files")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Create output subdirectories
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'raw').mkdir(exist_ok=True)
    (args.outdir / 'processed').mkdir(exist_ok=True)
    (args.outdir / 'pocket_times').mkdir(exist_ok=True)

    # Load model once
    model = load_model(args.checkpoint, args.config, device)

    # Gather test files
    test_files = sorted(args.test_dir.glob('[!.]*.sdf'))
    if args.test_list is not None:
        with open(args.test_list, 'r') as f:
            test_list = set(f.read().split(','))
        test_files = [x for x in test_files if x.stem in test_list]

    print(f"\nFound {len(test_files)} pockets in test set.")

    time_per_pocket = {}
    pbar = tqdm(test_files, desc="Pockets")

    for sdf_file in pbar:
        ligand_name = sdf_file.stem
        pdb_name, pocket_id, *suffix = ligand_name.split('_')
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")

        if not pdb_file.exists():
            warnings.warn(f"[SKIP] PDB not found for {ligand_name}: {pdb_file}")
            continue
        if not txt_file.exists():
            warnings.warn(f"[SKIP] TXT not found for {ligand_name}: {txt_file}")
            continue

        with open(txt_file, 'r') as f:
            resi_list = f.read().split()

        try:
            elapsed = generate_for_pocket(
                model=model,
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
            warnings.warn(f"[ERROR] Failed for {ligand_name}: {e}")
            continue

    # Save pocket times summary
    times_summary = args.outdir / 'pocket_times.txt'
    with open(times_summary, 'w') as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    if time_per_pocket:
        times_arr = torch.tensor(list(time_per_pocket.values()))
        print(f"\nTime per pocket: {times_arr.mean():.3f} +/- {times_arr.std(unbiased=False):.2f}s")
        print(f"Total pockets completed: {len(time_per_pocket)}")
        print(f"Output written to: {args.outdir}")


if __name__ == "__main__":
    main()