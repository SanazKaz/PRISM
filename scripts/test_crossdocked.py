"""
test.py — Generate ligands for a test set of pockets.

Scans a directory for pocket files and generates n_samples molecules per
pocket. Works with both DiffSBDD (--model diffsbdd) and TargetDiff
(--model targetdiff) checkpoints.

Expected layout under --test_dir:
    <test_dir>/
    ├── <stem>.sdf         reference ligand  (used as pocket name key)
    ├── <pdb_id>.pdb       pocket structure
    └── <stem>.txt         residue list (DiffSBDD only; optional — omit to
                           use the whole pocket)

For TargetDiff the .txt file is not used; only .pdb + .sdf are required.
This makes the script suitable for any custom test set of pockets.

Usage
-----
    # DiffSBDD (CrossDocked test set)
    python -m scripts.test \\
        checkpoints/my_run.ckpt \\
        --model diffsbdd \\
        --config configs/ppo_config.yaml \\
        --test_dir /data/crossdock/test \\
        --outdir results/test \\
        --n_samples 100 --batch_size 120 --sanitize

    # TargetDiff (any directory of .pdb + .sdf pairs)
    python -m scripts.test \\
        checkpoints/targetdiff.pt \\
        --model targetdiff \\
        --config configs/targetdiff_ppo.yaml \\
        --test_dir /data/custom_test \\
        --outdir results/test_td \\
        --n_samples 100 --batch_size 25
"""

import argparse
import traceback
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

_PROJECT_ROOT    = Path(__file__).resolve().parents[1]
_DIFFSBDD_ROOT   = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
_TARGETDIFF_ROOT = _PROJECT_ROOT / "src" / "models" / "targetdiff"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_DIFFSBDD_ROOT))
sys.path.insert(2, str(_TARGETDIFF_ROOT))

from src.prism.utils import dict_to_namespace, write_sdf_file
from src.prism.models.policy_factory import build_diffsbdd_policy
from src.prism.models.targetdiff_inference import (
    load_targetdiff_model, pocket_from_pdb, reconstruct_molecules,
    _ensure_targetdiff_utils,
)
from src.models.diffsbdd.analysis.molecule_builder import process_molecule

MAXITER  = 10
MAXNTRIES = 10


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_diffsbdd(checkpoint_path: Path, config_path: Path, device: str):
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


# ---------------------------------------------------------------------------
# Per-pocket generation — DiffSBDD
# ---------------------------------------------------------------------------

def generate_for_pocket_diffsbdd(
    ddpm_module,
    sdf_file: Path,
    pdb_file: Path,
    resi_list: list | None,
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
    """Generate n_samples molecules for one pocket with DiffSBDD. Returns elapsed seconds."""
    ligand_name   = sdf_file.stem
    raw_out       = outdir / "raw"       / f"{ligand_name}_gen.sdf"
    processed_out = outdir / "processed" / f"{ligand_name}_gen.sdf"
    time_file     = outdir / "pocket_times" / f"{ligand_name}.txt"

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
            all_molecules, valid_molecules, processed_molecules = [], [], []
            iter_count = 0

            while len(valid_molecules) < n_samples:
                iter_count += 1
                if iter_count > MAXITER:
                    raise RuntimeError("Maximum iterations exceeded.")

                num_nodes_batch = (
                    torch.ones(batch_size, dtype=torch.int) * num_nodes_lig
                    if num_nodes_lig is not None else None
                )
                with torch.no_grad():
                    mols_batch = ddpm_module.generate_ligands(
                        pdb_file, batch_size, resi_list,
                        num_nodes_lig=num_nodes_batch,
                        timesteps=timesteps,
                        sanitize=False, largest_frag=False, relax_iter=0,
                        n_nodes_bias=n_nodes_bias, n_nodes_min=n_nodes_min,
                        resamplings=resamplings, jump_length=jump_length,
                    )

                all_molecules.extend(mols_batch)
                batch_processed = [
                    process_molecule(m, sanitize=sanitize,
                                     relax_iter=(200 if relax else 0),
                                     largest_frag=not all_frags)
                    for m in mols_batch
                ]
                processed_molecules.extend(batch_processed)
                valid_molecules.extend(m for m in batch_processed if m is not None)

            valid_molecules = valid_molecules[:n_samples]
            all_molecules = (
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is not None] +
                [all_molecules[i] for i, m in enumerate(processed_molecules) if m is None]
            )
            write_sdf_file(raw_out, all_molecules)
            write_sdf_file(processed_out, valid_molecules)

            elapsed = time() - t_start
            time_file.write_text(f"{sdf_file} {elapsed}")
            return elapsed

        except (RuntimeError, ValueError) as e:
            if attempt >= MAXNTRIES - 1:
                raise RuntimeError(f"Max retries exceeded for {ligand_name}: {e}")
            warnings.warn(f"Attempt {attempt+1}/{MAXNTRIES} failed: {e}. Retrying…")

    return None


# ---------------------------------------------------------------------------
# Per-pocket generation — TargetDiff
# ---------------------------------------------------------------------------

def generate_for_pocket_targetdiff(
    model,
    protein_featurizer,
    sdf_file: Path,
    pdb_file: Path,
    outdir: Path,
    n_samples: int,
    batch_size: int,
    num_steps: int,
    center_pos_mode: str,
    skip_existing: bool,
) -> float | None:
    """Generate n_samples molecules for one pocket with TargetDiff. Returns elapsed seconds."""
    from scripts.sample_diffusion import sample_diffusion_ligand   # noqa: E402

    ligand_name   = sdf_file.stem
    processed_out = outdir / "processed" / f"{ligand_name}_gen.sdf"
    time_file     = outdir / "pocket_times" / f"{ligand_name}.txt"

    if skip_existing and time_file.exists() and processed_out.exists():
        with open(time_file) as f:
            return float(f.read().split()[1])

    t_start = time()
    device  = next(model.parameters()).device
    print(f"[DEBUG] {ligand_name} | device={device}  n_samples={n_samples}  "
          f"batch_size={batch_size}  num_batches={int(np.ceil(n_samples/batch_size))}  "
          f"num_steps={num_steps}")

    t0 = time()
    pocket_data = pocket_from_pdb(str(pdb_file), protein_featurizer)
    print(f"[DEBUG]   pocket loaded in {time()-t0:.2f}s  "
          f"| protein_pos={pocket_data.protein_pos.shape}  "
          f"| protein_feat={pocket_data.protein_atom_feature.shape}")

    t0 = time()
    with torch.no_grad():
        all_pred_pos, all_pred_v, *_ = sample_diffusion_ligand(
            model=model,
            data=pocket_data,
            num_samples=n_samples,
            batch_size=batch_size,
            device=str(device),
            num_steps=num_steps,
            pos_only=False,
            center_pos_mode=center_pos_mode,
            sample_num_atoms='prior',
        )
    t_sample = time() - t0
    print(f"[DEBUG]   sampling done in {t_sample:.2f}s ({t_sample/60:.1f}min)  "
          f"| {len(all_pred_pos)} mols  | {t_sample/max(len(all_pred_pos),1):.2f}s/mol")

    t0 = time()
    molecules = reconstruct_molecules(all_pred_pos, all_pred_v)
    t_recon = time() - t0
    valid = [m for m in molecules if m is not None]
    print(f"[DEBUG]   reconstruction done in {t_recon:.2f}s ({t_recon/60:.1f}min)  "
          f"| {len(valid)}/{len(molecules)} valid  "
          f"| {t_recon/max(len(molecules),1):.3f}s/mol")

    t0 = time()
    write_sdf_file(processed_out, valid)
    print(f"[DEBUG]   SDF written in {time()-t0:.2f}s -> {processed_out}")

    elapsed = time() - t_start
    time_file.write_text(f"{sdf_file} {elapsed}")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands for a test set of pockets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path,
                        help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--model", choices=["diffsbdd", "targetdiff"], default="diffsbdd",
                        help="Which model architecture the checkpoint belongs to")
    parser.add_argument("--config", type=Path, required=True,
                        help="PRISM YAML config")
    parser.add_argument("--test_dir", type=Path, required=True,
                        help="Directory containing .sdf, .pdb (and optionally .txt) files")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, default=None,
                        help="Optional comma-separated file of ligand stems to restrict generation")
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=120)
    # DiffSBDD options
    parser.add_argument("--sanitize",    action="store_true")
    parser.add_argument("--relax",       action="store_true")
    parser.add_argument("--all_frags",   action="store_true")
    parser.add_argument("--fix_n_nodes", action="store_true")
    parser.add_argument("--timesteps",   type=int, default=None,
                        help="Diffusion timesteps override (DiffSBDD)")
    parser.add_argument("--resamplings", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--n_nodes_bias", type=int, default=0)
    parser.add_argument("--n_nodes_min",  type=int, default=0)
    # TargetDiff options
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Diffusion steps (TargetDiff)")
    parser.add_argument("--center_pos_mode", type=str, default="protein",
                        help="Coordinate centering mode (TargetDiff)")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {args.model}")

    (args.outdir / "raw").mkdir(parents=True, exist_ok=True)
    (args.outdir / "processed").mkdir(exist_ok=True)
    (args.outdir / "pocket_times").mkdir(exist_ok=True)

    # --- Load model ---
    if args.model == "diffsbdd":
        diffsbdd_model = load_diffsbdd(args.checkpoint, args.config, device)
        td_model = td_featurizer = None
    elif args.model == "targetdiff":
        td_model = load_targetdiff_model(args.checkpoint, args.config, device)
        _ensure_targetdiff_utils()
        from utils import transforms as trans                          # noqa: E402
        td_featurizer = trans.FeaturizeProteinAtom()
        diffsbdd_model = None

    # --- Discover test pockets ---
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

        if not pdb_file.exists():
            warnings.warn(f"[SKIP] PDB not found: {pdb_file}")
            continue

        try:
            if args.model == "diffsbdd":
                txt_file  = sdf_file.parent / f"{ligand_name}.txt"
                resi_list = None
                if txt_file.exists():
                    with open(txt_file) as f:
                        resi_list = f.read().split()
                else:
                    warnings.warn(f"[INFO] No residue list for {ligand_name}; using full pocket.")

                elapsed = generate_for_pocket_diffsbdd(
                    ddpm_module=diffsbdd_model,
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

            elif args.model == "targetdiff":
                elapsed = generate_for_pocket_targetdiff(
                    model=td_model,
                    protein_featurizer=td_featurizer,
                    sdf_file=sdf_file,
                    pdb_file=pdb_file,
                    outdir=args.outdir,
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    num_steps=args.num_steps,
                    center_pos_mode=args.center_pos_mode,
                    skip_existing=args.skip_existing,
                )

            if elapsed is not None:
                time_per_pocket[str(sdf_file)] = elapsed
                pbar.set_description(f"Last: {ligand_name} ({elapsed:.1f}s)")

        except Exception as e:
            warnings.warn(f"[ERROR] {ligand_name}: {e}\n{traceback.format_exc()}")

    times_summary = args.outdir / "pocket_times.txt"
    with open(times_summary, "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    if time_per_pocket:
        times = list(time_per_pocket.values())
        print(f"\nTime per pocket: {np.mean(times):.1f} ± {np.std(times):.1f}s")
        print(f"Completed: {len(time_per_pocket)} pockets")
        print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
