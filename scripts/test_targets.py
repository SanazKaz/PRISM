"""
test_prism_targets.py

Generate 10,000 ligands for each of 6 PRISM evaluation targets.
Hardcoded paths for reproducible evaluation runs.
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

# --- Path Setup (same as generate_ligands.py) ---
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

MAXITER = 200  # Increased for 10k samples
MAXNTRIES = 5


# =============================================================================
# TEST TARGETS - Edit paths here if needed
# =============================================================================
BASE_PATH = Path("/data/stat-cadd/wolf7055/PRISM/data")


# =============================================================================
# TEST TARGETS - Edit paths here if needed
# =============================================================================
BASE_PATH = Path("/data/stat-cadd/wolf7055/PRISM/data")

TEST_TARGETS = {
    # BRD4_BD1 - 3 test structures
    "BRD4_BD1_4whw": {
        "pocket": BASE_PATH / "BRD4_BD1/02_preprocessed/pocket_files/4whw_3OT_B_201_pocket.pdb",
        "ligand": BASE_PATH / "BRD4_BD1/02_preprocessed/sdf_files/4whw_3OT_B_201.sdf",
    },
    "BRD4_BD1_6fo5": {
        "pocket": BASE_PATH / "BRD4_BD1/02_preprocessed/pocket_files/6fo5_DZH_B_201_pocket.pdb",
        "ligand": BASE_PATH / "BRD4_BD1/02_preprocessed/sdf_files/6fo5_DZH_B_201.sdf",
    },
    "BRD4_BD1_6xvc": {
        "pocket": BASE_PATH / "BRD4_BD1/02_preprocessed/pocket_files/6xvc_O32_D_203_pocket.pdb",
        "ligand": BASE_PATH / "BRD4_BD1/02_preprocessed/sdf_files/6xvc_O32_D_203.sdf",
    },
    
    # Carb_Anh_II - 3 test structures
    "Carb_Anh_II_6rl9": {
        "pocket": BASE_PATH / "Carb_Anh_II/02_preprocessed/pocket_files/6rl9_SAN_E_304_pocket.pdb",
        "ligand": BASE_PATH / "Carb_Anh_II/02_preprocessed/sdf_files/6rl9_SAN_E_304.sdf",
    },
    "Carb_Anh_II_3k34": {
        "pocket": BASE_PATH / "Carb_Anh_II/02_preprocessed/pocket_files/3k34_SUA_D_1003_pocket.pdb",
        "ligand": BASE_PATH / "Carb_Anh_II/02_preprocessed/sdf_files/3k34_SUA_D_1003.sdf",
    },
    "Carb_Anh_II_5n0d": {
        "pocket": BASE_PATH / "Carb_Anh_II/02_preprocessed/pocket_files/5n0d_8F2_C_302_pocket.pdb",
        "ligand": BASE_PATH / "Carb_Anh_II/02_preprocessed/sdf_files/5n0d_8F2_C_302.sdf",
    },
    
    # EGFR - 3 test structures
    "EGFR_8a27": {
        "pocket": BASE_PATH / "EGFR/02_preprocessed/pocket_files/8a27_KY9_C_1102_pocket.pdb",
        "ligand": BASE_PATH / "EGFR/02_preprocessed/sdf_files/8a27_KY9_C_1102.sdf",
    },
    "EGFR_3poz": {
        "pocket": BASE_PATH / "EGFR/02_preprocessed/pocket_files/3poz_03P_E_1023_pocket.pdb",
        "ligand": BASE_PATH / "EGFR/02_preprocessed/sdf_files/3poz_03P_E_1023.sdf",
    },
    "EGFR_4wkq": {
        "pocket": BASE_PATH / "EGFR/02_preprocessed/pocket_files/4wkq_IRE_B_1101_pocket.pdb",
        "ligand": BASE_PATH / "EGFR/02_preprocessed/sdf_files/4wkq_IRE_B_1101.sdf",
    },
    
    # Estrogen_recep_alpha - 3 test structures
    
    "Estrogen_recep_alpha_4ivy": {
        "pocket": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/pocket_files/4ivy_1GT_E_601_pocket.pdb",
        "ligand": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/sdf_files/4ivy_1GT_E_601.sdf",
    },

    "Estrogen_recep_alpha_5kct": {
        "pocket": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/pocket_files/5kct_OB6_F_601_pocket.pdb",
        "ligand": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/sdf_files/5kct_OB6_F_601.sdf",
    },
    "Estrogen_recep_alpha_2qzo": {
        "pocket": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/pocket_files/2qzo_KN1_E_1_pocket.pdb",
        "ligand": BASE_PATH / "Estrogen_recep_alpha/02_preprocessed/sdf_files/2qzo_KN1_E_1.sdf",
    },
    
    # Factor_Xa - 3 test structures
    "Factor_Xa_1ezq": {
        "pocket": BASE_PATH / "Factor_Xa/02_preprocessed/pocket_files/1ezq_RPR_D_265_pocket.pdb",
        "ligand": BASE_PATH / "Factor_Xa/02_preprocessed/sdf_files/1ezq_RPR_D_265.sdf",
    },
    "Factor_Xa_2p3t": {
        "pocket": BASE_PATH / "Factor_Xa/02_preprocessed/pocket_files/2p3t_993_E_500_pocket.pdb",
        "ligand": BASE_PATH / "Factor_Xa/02_preprocessed/sdf_files/2p3t_993_E_500.sdf",
    },
    "Factor_Xa_3kl6": {
        "pocket": BASE_PATH / "Factor_Xa/02_preprocessed/pocket_files/3kl6_443_C_1_pocket.pdb",
        "ligand": BASE_PATH / "Factor_Xa/02_preprocessed/sdf_files/3kl6_443_C_1.sdf",
    },
    
    # HIV_1_Protease - 3 test structures
    "HIV_1_Protease_2qnn": {
        "pocket": BASE_PATH / "HIV_1_Protease/02_preprocessed/pocket_files/2qnn_QN1_F_2501_pocket.pdb",
        "ligand": BASE_PATH / "HIV_1_Protease/02_preprocessed/sdf_files/2qnn_QN1_F_2501.sdf",
    },
    "HIV_1_Protease_3t11": {
        "pocket": BASE_PATH / "HIV_1_Protease/02_preprocessed/pocket_files/3t11_3T1_C_101_pocket.pdb",
        "ligand": BASE_PATH / "HIV_1_Protease/02_preprocessed/sdf_files/3t11_3T1_C_101.sdf",
    },
    "HIV_1_Protease_1hos": {
        "pocket": BASE_PATH / "HIV_1_Protease/02_preprocessed/pocket_files/1hos_PHP_C_400_pocket.pdb",
        "ligand": BASE_PATH / "HIV_1_Protease/02_preprocessed/sdf_files/1hos_PHP_C_400.sdf",
    },
}
# =============================================================================


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
    Initialize model from config and load weights from checkpoint.
    
    Args:
        checkpoint_path: Path to model weights (.pt or .ckpt)
        config_path: Path to model configuration YAML
        device: Device to load model onto ('cuda' or 'cpu')
    
    Returns:
        Loaded and initialized LigandPocketDDPM model
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
        raise RuntimeError(f"Model initialization failed: {e}")

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


def generate_for_target(
    model, 
    target_name: str,
    pocket_path: Path,
    ligand_path: Path,
    outdir: Path,
    n_samples: int = 10000,
    batch_size: int = 200,
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
    Generate molecules for a single target pocket.
    
    Args:
        model: Loaded LigandPocketDDPM model
        target_name: Name identifier for this target
        pocket_path: Path to pocket PDB file
        ligand_path: Path to reference ligand SDF
        outdir: Output directory
        n_samples: Number of valid molecules to generate
        batch_size: Generation batch size
        ... other generation parameters
    
    Returns:
        Dictionary with generation statistics
    """
    # Setup output paths
    target_outdir = outdir / target_name
    target_outdir.mkdir(parents=True, exist_ok=skip_existing)
    
    raw_sdf_file = target_outdir / f'{target_name}_raw.sdf'
    processed_sdf_file = target_outdir / f'{target_name}_processed.sdf'
    stats_file = target_outdir / f'{target_name}_stats.txt'
    
    # Check if already complete
    if skip_existing and processed_sdf_file.exists() and stats_file.exists():
        print(f"[SKIP] {target_name} already processed.")
        return None
    
    # Validate input files
    if not pocket_path.exists():
        raise FileNotFoundError(f"Pocket not found: {pocket_path}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand not found: {ligand_path}")
    
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} molecules for {target_name}")
    print(f"Pocket: {pocket_path.name}")
    print(f"Reference: {ligand_path.name}")
    print(f"{'='*60}")
    
    t_start = time()
    
    # Get reference ligand atom count if needed
    if fix_n_nodes:
        suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=False)
        if suppl[0] is None:
            raise ValueError(f"Could not read reference ligand: {ligand_path}")
        num_nodes_lig = suppl[0].GetNumHeavyAtoms()
        print(f"[INFO] Fixing atom count to {num_nodes_lig}")
    else:
        num_nodes_lig = None

    all_molecules = []
    valid_molecules = []
    processed_molecules = []
    iter_count = 0
    n_generated = 0
    n_valid = 0

    # Progress bar for this target
    pbar = tqdm(total=n_samples, desc=f"{target_name}", unit="mol")

    while len(valid_molecules) < n_samples:
        iter_count += 1
        if iter_count > MAXITER:
            warnings.warn(f"Max iterations reached for {target_name}. Got {len(valid_molecules)}/{n_samples}")
            break

        # Sample random atom counts for each molecule in batch (range 18-30)
        if num_nodes_lig is not None:
            # Fixed size from reference ligand
            num_nodes_lig_inflated = torch.ones(batch_size, dtype=int) * num_nodes_lig
            print(f"[INFO] ####################### num_nodes_lig_inflated: {num_nodes_lig_inflated}")
        else:
            # Random sampling from drug-like range
            num_nodes_lig_inflated = torch.randint(low=15, high=50, size=(batch_size,))
            print(f"[INFO] ####################### random sampling from drug-like range")
            print(f"[INFO] ####################### num_nodes_lig_inflated: {num_nodes_lig_inflated}")
        try:
            with torch.no_grad():
                mols_batch = model.generate_ligands(
                    pocket_path,
                    batch_size,
                    resi_list=None,
                    ref_ligand=str(ligand_path),
                    num_nodes_lig=num_nodes_lig_inflated,
                    timesteps=timesteps,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0,
                    n_nodes_bias=n_nodes_bias,
                    n_nodes_min=n_nodes_min,
                    resamplings=resamplings,
                    jump_length=jump_length
                )
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Batch generation failed: {e}")
            print(f"[ERROR] Pocket: {pocket_path}")
            print(f"[ERROR] Ligand: {ligand_path}")
            traceback.print_exc()
            continue

        all_molecules.extend(mols_batch)

        # Process molecules
        mols_batch_processed = [
            process_molecule(
                m,
                sanitize=sanitize,
                relax_iter=(200 if relax else 0),
                largest_frag=not all_frags
            )
            for m in mols_batch
        ]
        processed_molecules.extend(mols_batch_processed)
        
        valid_batch = [m for m in mols_batch_processed if m is not None]
        n_generated += batch_size
        n_valid += len(valid_batch)
        valid_molecules.extend(valid_batch)

        # Update progress
        pbar.n = min(len(valid_molecules), n_samples)
        pbar.set_postfix({
            'valid': f'{n_valid}/{n_generated}',
            'rate': f'{n_valid/n_generated*100:.1f}%' if n_generated > 0 else 'N/A'
        })
        pbar.refresh()

    pbar.close()

    # Trim to exact count
    valid_molecules = valid_molecules[:n_samples]

    # Reorder: valid first, then invalid
    all_molecules = \
        [all_molecules[i] for i, m in enumerate(processed_molecules) if m is not None] + \
        [all_molecules[i] for i, m in enumerate(processed_molecules) if m is None]

    # Save molecules
    print(f"[SAVE] Writing {len(all_molecules)} raw molecules...")
    utils.write_sdf_file(raw_sdf_file, all_molecules)
    
    print(f"[SAVE] Writing {len(valid_molecules)} processed molecules...")
    utils.write_sdf_file(processed_sdf_file, valid_molecules)

    # Calculate stats
    elapsed = time() - t_start
    validity_rate = n_valid / n_generated if n_generated > 0 else 0
    
    stats = {
        'target': target_name,
        'n_valid': len(valid_molecules),
        'n_generated': n_generated,
        'validity_rate': validity_rate,
        'time_seconds': elapsed,
        'time_per_mol': elapsed / len(valid_molecules) if valid_molecules else 0,
    }

    # Save stats
    with open(stats_file, 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"[DONE] {target_name}: {len(valid_molecules)} valid molecules")
    print(f"       Validity: {validity_rate*100:.2f}% | Time: {elapsed/60:.1f} min")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10k ligands for each PRISM evaluation target"
    )
    parser.add_argument('checkpoint', type=Path,
                        help="Path to model weights (.pt or .ckpt)")
    parser.add_argument('--config', type=Path, required=True,
                        help="Path to model config YAML")
    parser.add_argument('--outdir', type=Path, required=True,
                        help="Output directory")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help="Number of valid molecules per target (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=200,
                        help="Batch size for generation (default: 200)")
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help="Specific targets to run (default: all)")
    parser.add_argument('--target', type=str, default=None,
                        help="Single target to run (for parallel job submission)")
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
                        help="Skip targets that are already complete")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Load model once
    model = load_model(args.checkpoint, args.config, device)

    # Select targets (--target takes precedence for single-target parallel jobs)
    if args.target:
        if args.target not in TEST_TARGETS:
            print(f"[ERROR] Unknown target: {args.target}")
            print(f"Available: {list(TEST_TARGETS.keys())}")
            return
        targets = {args.target: TEST_TARGETS[args.target]}
    elif args.targets:
        targets = {k: v for k, v in TEST_TARGETS.items() if k in args.targets}
        if not targets:
            print(f"[ERROR] No matching targets. Available: {list(TEST_TARGETS.keys())}")
            return
    else:
        targets = TEST_TARGETS

    print(f"\nWill generate {args.n_samples} molecules for {len(targets)} targets:")
    for name in targets:
        print(f"  - {name}")

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Generate for each target
    all_stats = []
    for target_name, paths in targets.items():
        try:
            stats = generate_for_target(
                model=model,
                target_name=target_name,
                pocket_path=paths['pocket'],
                ligand_path=paths['ligand'],
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
            if stats:
                all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] Failed for {target_name}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    
    if all_stats:
        total_mols = sum(s['n_valid'] for s in all_stats)
        total_time = sum(s['time_seconds'] for s in all_stats)
        avg_validity = np.mean([s['validity_rate'] for s in all_stats])
        
        print(f"Total molecules: {total_mols}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average validity: {avg_validity*100:.2f}%")
        
        # Save summary
        summary_file = args.outdir / 'generation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"N samples per target: {args.n_samples}\n\n")
            for s in all_stats:
                f.write(f"{s['target']}: {s['n_valid']} mols, "
                       f"{s['validity_rate']*100:.2f}% valid, "
                       f"{s['time_seconds']/60:.1f} min\n")
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()