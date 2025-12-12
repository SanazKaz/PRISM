"""
generate_ligands.py

Standalone script for generating ligands using a PPO finetuned DiffSBDD model.
Supports both clean .pt files and Lightning .ckpt files (automatically handling prefixes).
"""

import argparse
import sys
import os
import inspect
import yaml
from pathlib import Path
from argparse import Namespace

import torch
import numpy as np
from openbabel import openbabel

# Suppress OpenBabel logging
openbabel.obErrorLog.StopLogging()

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
DIFFSBDD_PATH = Path(PROJECT_ROOT) / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(DIFFSBDD_PATH))

# --- Imports ---
try:
    from lightning_modules import LigandPocketDDPM
    import utils
except ImportError:
    from src.models.diffsbdd.lightning_modules import LigandPocketDDPM
    from src.models.diffsbdd import utils


class RecursiveNamespace(Namespace):
    """Recursively converts a dictionary to a Namespace."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                setattr(self, k, RecursiveNamespace(**v))
            elif isinstance(v, list):
                setattr(self, k, [RecursiveNamespace(**i) if isinstance(i, dict) else i for i in v])


def load_model(args, device):
    """Initializes model architecture and loads weights from .pt or .ckpt."""
    
    # 1. Load Configuration
    with open(args.config, 'r') as f:
        raw_config = yaml.safe_load(f)
    full_config_ns = RecursiveNamespace(**raw_config)

    # 2. Filter Configuration Arguments
    sig = inspect.signature(LigandPocketDDPM.__init__)
    valid_keys = set(sig.parameters.keys()) - {'self'}

    if hasattr(full_config_ns, 'model') and isinstance(full_config_ns.model, Namespace):
        source_params = full_config_ns.model.__dict__
    else:
        source_params = full_config_ns.__dict__

    filtered_args = {k: v for k, v in source_params.items() if k in valid_keys}

    # 3. Load Node Histogram
    if 'datadir' in filtered_args:
        datadir = Path(filtered_args['datadir'])
        histogram_path = datadir / 'size_distribution.npy'
        if histogram_path.exists():
            filtered_args['node_histogram'] = np.load(histogram_path).tolist()
        else:
            print(f"[WARNING] Size distribution not found at {histogram_path}. Initialization may fail.")
    
    # 4. Initialize Model Architecture
    try:
        model = LigandPocketDDPM(**filtered_args)
    except TypeError as e:
        print(f"[ERROR] Model initialization failed: {e}")
        sys.exit(1)

    # 5. Load Weights (Universal Handler)
    print(f"Loading weights from {args.model_path}...")
    # weights_only=False needed for some complex objects
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Detect if it is a Lightning Checkpoint (.ckpt)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("[INFO] Detected Lightning checkpoint format.")
        state_dict = checkpoint['state_dict']
        
        # Clean up PPO wrapper prefixes (remove 'ddpm_model.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ddpm_model.'):
                new_state_dict[k.replace('ddpm_model.', '')] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    else:
        # Assume it is a direct .pt dump
        print("[INFO] Detected raw state_dict format.")
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
        print("[SUCCESS] Weights loaded successfully.")
    except RuntimeError:
        print("[INFO] Standard load failed (key mismatch). Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    model = load_model(args, device)

    # --- Setup Generation Parameters ---
    if args.batch_size is None:
        args.batch_size = args.n_samples
    
    if args.save_traj and args.n_samples > 1:
        print("[INFO] Trajectory saving is only supported for n_samples=1. Setting n_samples=1.")
        args.n_samples = 1

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * args.num_nodes_lig
    else:
        num_nodes_lig = None

    molecules = []
    num_batches = int(np.ceil(args.n_samples / args.batch_size))
    
    print(f"Starting generation of {args.n_samples} molecules...")

    # --- Generation Loop ---
    for i in range(num_batches):
        current_batch_size = min(args.batch_size, args.n_samples - len(molecules))
        
        if num_nodes_lig is not None:
            batch_num_nodes = num_nodes_lig[len(molecules) : len(molecules) + current_batch_size]
        else:
            batch_num_nodes = None

        with torch.no_grad():
            molecules_batch = model.generate_ligands(
                args.pdbfile, 
                current_batch_size, 
                args.resi_list, 
                args.ref_ligand,
                batch_num_nodes,  # <--- Pass the sliced batch, not the full list
                args.sanitize, 
                largest_frag=not args.all_frags,
                relax_iter=(200 if args.relax else 0),
                resamplings=args.resamplings, 
                jump_length=args.jump_length,
                timesteps=args.timesteps, 
                save_traj=args.save_traj
            )
        molecules.extend(molecules_batch)
        print(f"Batch {i+1}/{num_batches} complete. Total generated: {len(molecules)}")
        
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sdf_file(args.outfile, molecules)
    print(f"Successfully saved {len(molecules)} molecules to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path, help="Path to weights (.pt or .ckpt)")
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--pdbfile', type=str, required=True)
    parser.add_argument('--outfile', type=Path, required=True)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--save_traj', action='store_true')
    
    args = parser.parse_args()
    main(args)