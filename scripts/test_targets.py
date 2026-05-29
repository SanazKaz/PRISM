"""
test_targets.py — Generate ligands for PRISM's held-out evaluation targets.

Loops over a fixed set of six protein targets (18 structures total) and
generates n_samples valid molecules per pocket. Works with both DiffSBDD
(--model diffsbdd) and TargetDiff (--model targetdiff) checkpoints.

Usage
-----
    python -m scripts.test_targets \\
        checkpoints/my_run.ckpt \\
        --model diffsbdd \\
        --config configs/ppo_config.yaml \\
        --targets_dir /data/my_project/data \\
        --outdir results/test_targets \\
        --n_samples 10000 --batch_size 200 --sanitize

Run a single target (useful for parallel job submission):
    python -m scripts.test_targets ... --target BRD4_BD1_4whw

The expected directory layout under --targets_dir is:
    <targets_dir>/
    └── <TargetName>/
        └── 02_preprocessed/
            ├── pocket_files/<pdb>_<lig>_<chain>_<resid>_pocket.pdb
            └── sdf_files/<pdb>_<lig>_<chain>_<resid>.sdf
"""

import argparse
import warnings
import sys
from pathlib import Path
from time import time

import torch
import numpy as np
import yaml
from rdkit import Chem
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
)
from src.models.diffsbdd.analysis.molecule_builder import process_molecule


# ---------------------------------------------------------------------------
# Target definitions — paths are relative to --targets_dir
# ---------------------------------------------------------------------------

_TARGET_SPECS = {
    "BRD4_BD1_4whw":            ("BRD4_BD1",          "4whw_3OT_B_201"),
    "BRD4_BD1_6fo5":            ("BRD4_BD1",          "6fo5_DZH_B_201"),
    "BRD4_BD1_6xvc":            ("BRD4_BD1",          "6xvc_O32_D_203"),
    "Carb_Anh_II_6rl9":         ("Carb_Anh_II",       "6rl9_SAN_E_304"),
    "Carb_Anh_II_3k34":         ("Carb_Anh_II",       "3k34_SUA_D_1003"),
    "Carb_Anh_II_5n0d":         ("Carb_Anh_II",       "5n0d_8F2_C_302"),
    "EGFR_8a27":                ("EGFR",              "8a27_KY9_C_1102"),
    "EGFR_3poz":                ("EGFR",              "3poz_03P_E_1023"),
    "EGFR_4wkq":                ("EGFR",              "4wkq_IRE_B_1101"),
    "Estrogen_recep_alpha_4ivy": ("Estrogen_recep_alpha", "4ivy_1GT_E_601"),
    "Estrogen_recep_alpha_5kct": ("Estrogen_recep_alpha", "5kct_OB6_F_601"),
    "Estrogen_recep_alpha_2qzo": ("Estrogen_recep_alpha", "2qzo_KN1_E_1"),
    "Factor_Xa_1ezq":           ("Factor_Xa",         "1ezq_RPR_D_265"),
    "Factor_Xa_2p3t":           ("Factor_Xa",         "2p3t_993_E_500"),
    "Factor_Xa_3kl6":           ("Factor_Xa",         "3kl6_443_C_1"),
    "HIV_1_Protease_2qnn":      ("HIV_1_Protease",    "2qnn_QN1_F_2501"),
    "HIV_1_Protease_3t11":      ("HIV_1_Protease",    "3t11_3T1_C_101"),
    "HIV_1_Protease_1hos":      ("HIV_1_Protease",    "1hos_PHP_C_400"),
}

MAXITER = 200


def _build_targets(targets_dir: Path) -> dict:
    """Resolve _TARGET_SPECS into absolute pocket/ligand paths under targets_dir."""
    targets = {}
    for key, (protein, basename) in _TARGET_SPECS.items():
        base = targets_dir / protein / "02_preprocessed"
        targets[key] = {
            "pocket": base / "pocket_files" / f"{basename}_pocket.pdb",
            "ligand": base / "sdf_files"   / f"{basename}.sdf",
        }
    return targets


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
# Per-target generation — DiffSBDD
# ---------------------------------------------------------------------------

def generate_for_target_diffsbdd(
    ddpm_module,
    target_name: str,
    pocket_path: Path,
    ligand_path: Path,
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
) -> dict | None:
    target_outdir = outdir / target_name
    target_outdir.mkdir(parents=True, exist_ok=True)

    raw_sdf       = target_outdir / f"{target_name}_raw.sdf"
    processed_sdf = target_outdir / f"{target_name}_processed.sdf"
    stats_file    = target_outdir / f"{target_name}_stats.txt"

    if skip_existing and processed_sdf.exists() and stats_file.exists():
        print(f"[SKIP] {target_name} — output already exists.")
        return None

    if not pocket_path.exists():
        raise FileNotFoundError(f"Pocket not found: {pocket_path}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand not found: {ligand_path}")

    print(f"\n{'='*60}\nTarget: {target_name}\nPocket: {pocket_path.name}\n{'='*60}")

    if fix_n_nodes:
        suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=False)
        ref_mol = suppl[0]
        if ref_mol is None:
            raise ValueError(f"Could not read reference ligand: {ligand_path}")
        num_nodes_lig = ref_mol.GetNumHeavyAtoms()
        print(f"Fixing atom count to reference: {num_nodes_lig}")
    else:
        num_nodes_lig = None

    t_start = time()
    all_molecules, valid_molecules, processed_molecules = [], [], []
    n_generated = n_valid = 0

    pbar = tqdm(total=n_samples, desc=target_name, unit="mol")
    iter_count = 0

    while len(valid_molecules) < n_samples:
        iter_count += 1
        if iter_count > MAXITER:
            warnings.warn(f"Max iterations for {target_name} "
                          f"({len(valid_molecules)}/{n_samples} valid).")
            break

        num_nodes_batch = (
            torch.ones(batch_size, dtype=torch.int) * num_nodes_lig
            if num_nodes_lig is not None else None
        )
        try:
            with torch.no_grad():
                mols_batch = ddpm_module.generate_ligands(
                    pocket_path, batch_size,
                    resi_list=None,
                    ref_ligand=str(ligand_path),
                    num_nodes_lig=num_nodes_batch,
                    timesteps=timesteps,
                    sanitize=False, largest_frag=False, relax_iter=0,
                    n_nodes_bias=n_nodes_bias, n_nodes_min=n_nodes_min,
                    resamplings=resamplings, jump_length=jump_length,
                )
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Batch failed for {target_name}: {e}")
            traceback.print_exc()
            continue

        all_molecules.extend(mols_batch)
        batch_processed = [
            process_molecule(m, sanitize=sanitize,
                             relax_iter=(200 if relax else 0),
                             largest_frag=not all_frags)
            for m in mols_batch
        ]
        processed_molecules.extend(batch_processed)
        valid_batch = [m for m in batch_processed if m is not None]
        n_generated += batch_size
        n_valid     += len(valid_batch)
        valid_molecules.extend(valid_batch)

        pbar.n = min(len(valid_molecules), n_samples)
        pbar.set_postfix({
            "valid": f"{n_valid}/{n_generated}",
            "rate":  f"{n_valid/n_generated*100:.1f}%" if n_generated else "N/A",
        })
        pbar.refresh()

    pbar.close()
    valid_molecules = valid_molecules[:n_samples]

    all_molecules = (
        [all_molecules[i] for i, m in enumerate(processed_molecules) if m is not None] +
        [all_molecules[i] for i, m in enumerate(processed_molecules) if m is None]
    )
    write_sdf_file(raw_sdf, all_molecules)
    write_sdf_file(processed_sdf, valid_molecules)

    elapsed = time() - t_start
    validity_rate = n_valid / n_generated if n_generated else 0

    stats = {"target": target_name, "n_valid": len(valid_molecules),
             "n_generated": n_generated, "validity_rate": validity_rate,
             "time_seconds": elapsed}
    with open(stats_file, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"[DONE] {target_name}: {len(valid_molecules)} valid | "
          f"validity {validity_rate*100:.1f}% | {elapsed/60:.1f} min")
    return stats


# ---------------------------------------------------------------------------
# Per-target generation — TargetDiff
# ---------------------------------------------------------------------------

def generate_for_target_targetdiff(
    model,
    protein_featurizer,
    target_name: str,
    pocket_path: Path,
    ligand_path: Path,
    outdir: Path,
    n_samples: int,
    batch_size: int,
    num_steps: int,
    center_pos_mode: str,
    skip_existing: bool,
) -> dict | None:
    from scripts.sample_diffusion import sample_diffusion_ligand   # noqa: E402

    target_outdir = outdir / target_name
    target_outdir.mkdir(parents=True, exist_ok=True)

    processed_sdf = target_outdir / f"{target_name}_processed.sdf"
    stats_file    = target_outdir / f"{target_name}_stats.txt"

    if skip_existing and processed_sdf.exists() and stats_file.exists():
        print(f"[SKIP] {target_name} — output already exists.")
        return None

    if not pocket_path.exists():
        raise FileNotFoundError(f"Pocket not found: {pocket_path}")

    print(f"\n{'='*60}\nTarget: {target_name}\nPocket: {pocket_path.name}\n{'='*60}")

    t_start = time()
    device  = next(model.parameters()).device

    pocket_data = pocket_from_pdb(str(pocket_path), protein_featurizer)

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

    molecules     = reconstruct_molecules(all_pred_pos, all_pred_v)
    valid         = [m for m in molecules if m is not None]
    elapsed       = time() - t_start
    validity_rate = len(valid) / len(molecules) if molecules else 0

    write_sdf_file(processed_sdf, valid)

    stats = {"target": target_name, "n_valid": len(valid),
             "n_generated": len(molecules), "validity_rate": validity_rate,
             "time_seconds": elapsed}
    with open(stats_file, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"[DONE] {target_name}: {len(valid)} valid | "
          f"validity {validity_rate*100:.1f}% | {elapsed/60:.1f} min")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands for PRISM evaluation targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path,
                        help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--model", choices=["diffsbdd", "targetdiff"], default="diffsbdd",
                        help="Which model architecture the checkpoint belongs to")
    parser.add_argument("--config", type=Path, required=True,
                        help="PRISM YAML config")
    parser.add_argument("--targets_dir", type=Path, required=True,
                        help="Root data directory containing per-target subdirectories")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--target",  type=str, default=None,
                        help="Run a single target by key")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                        help="Subset of target keys to run")
    # DiffSBDD options
    parser.add_argument("--sanitize",    action="store_true")
    parser.add_argument("--relax",       action="store_true")
    parser.add_argument("--all_frags",   action="store_true")
    parser.add_argument("--fix_n_nodes", action="store_true")
    parser.add_argument("--timesteps",   type=int, default=None)
    parser.add_argument("--resamplings", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--n_nodes_bias", type=int, default=0)
    parser.add_argument("--n_nodes_min",  type=int, default=0)
    # TargetDiff options
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--center_pos_mode", type=str, default="protein")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {args.model}")

    if args.model == "diffsbdd":
        diffsbdd_model = load_diffsbdd(args.checkpoint, args.config, device)
        td_model = td_featurizer = None
    elif args.model == "targetdiff":
        td_model = load_targetdiff_model(args.checkpoint, args.config, device)
        from utils import transforms as trans                          # noqa: E402
        td_featurizer  = trans.FeaturizeProteinAtom()
        diffsbdd_model = None

    all_targets = _build_targets(args.targets_dir)

    if args.target:
        if args.target not in all_targets:
            print(f"[ERROR] Unknown target '{args.target}'. "
                  f"Available: {list(all_targets.keys())}")
            return
        targets = {args.target: all_targets[args.target]}
    elif args.targets:
        targets = {k: v for k, v in all_targets.items() if k in args.targets}
    else:
        targets = all_targets

    print(f"\nRunning {len(targets)} target(s): {list(targets.keys())}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for name, paths in targets.items():
        try:
            if args.model == "diffsbdd":
                stats = generate_for_target_diffsbdd(
                    ddpm_module=diffsbdd_model,
                    target_name=name,
                    pocket_path=paths["pocket"],
                    ligand_path=paths["ligand"],
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
                stats = generate_for_target_targetdiff(
                    model=td_model,
                    protein_featurizer=td_featurizer,
                    target_name=name,
                    pocket_path=paths["pocket"],
                    ligand_path=paths["ligand"],
                    outdir=args.outdir,
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    num_steps=args.num_steps,
                    center_pos_mode=args.center_pos_mode,
                    skip_existing=args.skip_existing,
                )

            if stats:
                all_stats.append(stats)

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue

    if all_stats:
        total_mols   = sum(s["n_valid"] for s in all_stats)
        total_time   = sum(s["time_seconds"] for s in all_stats)
        avg_validity = np.mean([s["validity_rate"] for s in all_stats])

        print(f"\n{'='*60}")
        print(f"COMPLETE — {total_mols} molecules across {len(all_stats)} target(s)")
        print(f"Average validity: {avg_validity*100:.1f}%  |  "
              f"Total time: {total_time/60:.1f} min")

        summary = args.outdir / "generation_summary.txt"
        with open(summary, "w") as f:
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"config:     {args.config}\n")
            f.write(f"model:      {args.model}\n")
            f.write(f"n_samples:  {args.n_samples}\n\n")
            for s in all_stats:
                f.write(f"{s['target']}: {s['n_valid']} mols, "
                        f"{s['validity_rate']*100:.1f}% valid, "
                        f"{s['time_seconds']/60:.1f} min\n")
        print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
