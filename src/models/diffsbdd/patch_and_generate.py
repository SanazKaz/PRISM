#!/usr/bin/env python3
"""
Patch checkpoint to add missing parameters and run generation.
This script handles AttributeErrors for missing parameters in older checkpoints.
"""

import torch
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace


def patch_checkpoint(checkpoint_path, num_nodes_lig=20, ligand_chunk_size=10, top_k=50,
                    progressive_training=True, start_timesteps=20, end_timesteps=126, 
                    progressive_training_epochs=100):
    """
    Load checkpoint and add missing parameters if not present.
    
    Args:
        checkpoint_path (str): Path to the original checkpoint file
        num_nodes_lig (int): Number of ligand nodes to add to config
        ligand_chunk_size (int): Ligand chunk size to add to config
        top_k (int): Top k parameter for PPO
        progressive_training (bool): Whether to use progressive training
        start_timesteps (int): Starting timesteps for progressive training
        end_timesteps (int): Ending timesteps for progressive training
        progressive_training_epochs (int): Number of epochs for progressive training
        
    Returns:
        str: Path to the patched checkpoint file
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Check and patch the configuration
    patched = False
    
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        print(f"Found hyper_parameters: {type(hyper_params)}")
        
        # Handle both dictionary and object structures
        if isinstance(hyper_params, dict):
            # Dictionary structure
            if 'outdir' in hyper_params and not isinstance(hyper_params['outdir'], str):
                hyper_params['outdir'] = str(hyper_params['outdir'])
                patched = True
                print(f"✓ Added outdir={hyper_params['outdir']} to hyper_parameters dict")
            else:
                if hasattr(hyper_params, 'outdir') and not isinstance(hyper_params.outdir, str):
                    hyper_params.outdir = str(hyper_params.outdir)
                    patched = True
                    print(f"✓ Converted outdir to string: {hyper_params.outdir}")
                
            if 'ppo_params' in hyper_params:
                ppo_params = hyper_params['ppo_params']
                if isinstance(ppo_params, dict):
                    # Check and add num_nodes_lig
                    if 'num_nodes_lig' not in ppo_params:
                        ppo_params['num_nodes_lig'] = num_nodes_lig
                        patched = True
                        print(f"✓ Added num_nodes_lig={num_nodes_lig} to ppo_params dict")
                    else:
                        print(f"✓ num_nodes_lig already exists: {ppo_params['num_nodes_lig']}")
                    
                    # Check and add ligand_chunk_size
                    if 'ligand_chunk_size' not in ppo_params:
                        ppo_params['ligand_chunk_size'] = ligand_chunk_size
                        patched = True
                        print(f"✓ Added ligand_chunk_size={ligand_chunk_size} to ppo_params dict")
                    else:
                        print(f"✓ ligand_chunk_size already exists: {ppo_params['ligand_chunk_size']}")
                    
                    # Check and add top_k
                    if 'top_k' not in ppo_params:
                        ppo_params['top_k'] = top_k
                        patched = True
                        print(f"✓ Added top_k={top_k} to ppo_params dict")
                    else:
                        print(f"✓ top_k already exists: {ppo_params['top_k']}")
                        
                    # Check and add progressive training parameters
                    if 'progressive_training_epochs' not in ppo_params:
                        ppo_params['progressive_training_epochs'] = progressive_training_epochs
                        patched = True
                        print(f"✓ Added progressive_training_epochs={progressive_training_epochs} to ppo_params dict")
                    else:
                        print(f"✓ progressive_training_epochs already exists: {ppo_params['progressive_training_epochs']}")
                    
                    if 'start_timesteps' not in ppo_params:
                        ppo_params['start_timesteps'] = start_timesteps
                        patched = True
                        print(f"✓ Added start_timesteps={start_timesteps} to ppo_params dict")
                    else:
                        print(f"✓ start_timesteps already exists: {ppo_params['start_timesteps']}")
                    
                    if 'end_timesteps' not in ppo_params:
                        ppo_params['end_timesteps'] = end_timesteps
                        patched = True
                        print(f"✓ Added end_timesteps={end_timesteps} to ppo_params dict")
                    else:
                        print(f"✓ end_timesteps already exists: {ppo_params['end_timesteps']}")
                    
                    if 'progressive_training' not in ppo_params:
                        ppo_params['progressive_training'] = progressive_training
                        patched = True
                        print(f"✓ Added progressive_training={progressive_training} to ppo_params dict")
                    else:
                        print(f"✓ progressive_training already exists: {ppo_params['progressive_training']}")
                else:
                    # ppo_params is an object
                    if not hasattr(ppo_params, 'num_nodes_lig'):
                        ppo_params.num_nodes_lig = num_nodes_lig
                        patched = True
                        print(f"✓ Added num_nodes_lig={num_nodes_lig} to ppo_params object")
                    else:
                        print(f"✓ num_nodes_lig already exists: {ppo_params.num_nodes_lig}")
                    
                    if not hasattr(ppo_params, 'ligand_chunk_size'):
                        ppo_params.ligand_chunk_size = ligand_chunk_size
                        patched = True
                        print(f"✓ Added ligand_chunk_size={ligand_chunk_size} to ppo_params object")
                    else:
                        print(f"✓ ligand_chunk_size already exists: {ppo_params.ligand_chunk_size}")
                    
                    if not hasattr(ppo_params, 'top_k'):
                        ppo_params.top_k = top_k
                        patched = True
                        print(f"✓ Added top_k={top_k} to ppo_params object")
                    else:
                        print(f"✓ top_k already exists: {ppo_params.top_k}")
                    
                    if not hasattr(ppo_params, 'progressive_training_epochs'):
                        ppo_params.progressive_training_epochs = progressive_training_epochs
                        patched = True
                        print(f"✓ Added progressive_training_epochs={progressive_training_epochs} to ppo_params object")
                    else:
                        print(f"✓ progressive_training_epochs already exists: {ppo_params.progressive_training_epochs}")
                    
                    if not hasattr(ppo_params, 'start_timesteps'):
                        ppo_params.start_timesteps = start_timesteps
                        patched = True
                        print(f"✓ Added start_timesteps={start_timesteps} to ppo_params object")
                    else:
                        print(f"✓ start_timesteps already exists: {ppo_params.start_timesteps}")
                    
                    if not hasattr(ppo_params, 'end_timesteps'):
                        ppo_params.end_timesteps = end_timesteps
                        patched = True
                        print(f"✓ Added end_timesteps={end_timesteps} to ppo_params object")
                    else:
                        print(f"✓ end_timesteps already exists: {ppo_params.end_timesteps}")
                    
                    if not hasattr(ppo_params, 'progressive_training'):
                        ppo_params.progressive_training = progressive_training
                        patched = True
                        print(f"✓ Added progressive_training={progressive_training} to ppo_params object")
                    else:
                        print(f"✓ progressive_training already exists: {ppo_params.progressive_training}")
            else:
                print("⚠ ppo_params not found in hyper_parameters dict")
                # Create ppo_params as a dict
                hyper_params['ppo_params'] = {
                    'num_nodes_lig': num_nodes_lig,
                    'ligand_chunk_size': ligand_chunk_size,
                    'top_k': top_k,
                    'progressive_training': progressive_training,
                    'start_timesteps': start_timesteps,
                    'end_timesteps': end_timesteps,
                    'progressive_training_epochs': progressive_training_epochs
                }
                patched = True
                print(f"✓ Created ppo_params dict with all missing parameters")
        else:
            # Object structure
            if hasattr(hyper_params, 'ppo_params'):
                ppo_params = hyper_params.ppo_params
                if not hasattr(ppo_params, 'num_nodes_lig'):
                    ppo_params.num_nodes_lig = num_nodes_lig
                    patched = True
                    print(f"✓ Added num_nodes_lig={num_nodes_lig} to ppo_params")
                else:
                    print(f"✓ num_nodes_lig already exists: {ppo_params.num_nodes_lig}")
                
                if not hasattr(ppo_params, 'ligand_chunk_size'):
                    ppo_params.ligand_chunk_size = ligand_chunk_size
                    patched = True
                    print(f"✓ Added ligand_chunk_size={ligand_chunk_size} to ppo_params")
                else:
                    print(f"✓ ligand_chunk_size already exists: {ppo_params.ligand_chunk_size}")
                
                if not hasattr(ppo_params, 'top_k'):
                    ppo_params.top_k = top_k
                    patched = True
                    print(f"✓ Added top_k={top_k} to ppo_params")
                else:
                    print(f"✓ top_k already exists: {ppo_params.top_k}")
                
                if not hasattr(ppo_params, 'progressive_training_epochs'):
                    ppo_params.progressive_training_epochs = progressive_training_epochs
                    patched = True
                    print(f"✓ Added progressive_training_epochs={progressive_training_epochs} to ppo_params object")
                else:
                    print(f"✓ progressive_training_epochs already exists: {ppo_params.progressive_training_epochs}")
                
                if not hasattr(ppo_params, 'start_timesteps'):
                    ppo_params.start_timesteps = start_timesteps
                    patched = True
                    print(f"✓ Added start_timesteps={start_timesteps} to ppo_params object")
                else:
                    print(f"✓ start_timesteps already exists: {ppo_params.start_timesteps}")
                
                if not hasattr(ppo_params, 'end_timesteps'):
                    ppo_params.end_timesteps = end_timesteps
                    patched = True
                    print(f"✓ Added end_timesteps={end_timesteps} to ppo_params object")
                else:
                    print(f"✓ end_timesteps already exists: {ppo_params.end_timesteps}")
                
                if not hasattr(ppo_params, 'progressive_training'):
                    ppo_params.progressive_training = progressive_training
                    patched = True
                    print(f"✓ Added progressive_training={progressive_training} to ppo_params object")
                else:
                    print(f"✓ progressive_training already exists: {ppo_params.progressive_training}")
            else:
                print("⚠ ppo_params not found in hyper_parameters object")
                # Create ppo_params if it doesn't exist
                hyper_params.ppo_params = SimpleNamespace()
                hyper_params.ppo_params.num_nodes_lig = num_nodes_lig
                hyper_params.ppo_params.ligand_chunk_size = ligand_chunk_size
                hyper_params.ppo_params.top_k = top_k
                hyper_params.ppo_params.progressive_training = progressive_training
                hyper_params.ppo_params.start_timesteps = start_timesteps
                hyper_params.ppo_params.end_timesteps = end_timesteps
                hyper_params.ppo_params.progressive_training_epochs = progressive_training_epochs
                patched = True
                print(f"✓ Created ppo_params with all missing parameters")
    else:
        print("⚠ No hyper_parameters found in checkpoint")
        sys.exit(1)
    
    # Save patched checkpoint
    if patched:
        patched_path = checkpoint_path.replace('.ckpt', '_patched.ckpt')
        torch.save(checkpoint, patched_path)
        print(f"✓ Saved patched checkpoint to: {patched_path}")
        return patched_path
    else:
        print("✓ No patching needed, using original checkpoint")
        return checkpoint_path

def run_generation(checkpoint_path, args):
    """
    Run the generation command with the specified parameters.
    
    Args:
        checkpoint_path (str): Path to the (possibly patched) checkpoint
        args (SimpleNamespace): Parameters for generation
    """
    
    # Build the generation command
    cmd = [
        'python', 'src/models/diffsbdd/generate_ligands.py',
        checkpoint_path,
        '--pdbfile', args.pdbfile,
        '--outfile', args.outfile,
        '--ref_ligand', args.ref_ligand,
        '--n_samples', str(args.n_samples),
        '--timesteps', str(args.timesteps),
        '--num_nodes_lig', str(args.num_nodes_lig)
    ]
    
    # **ADD THIS BLOCK TO PASS THE FLAG**
    if args.save_traj:
        cmd.append('--save_traj')
    
    print("\n" + "="*60)
    print("RUNNING GENERATION COMMAND:")
    print(" ".join(cmd))
    print("="*60 + "\n")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Generation completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Generation failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n✗ Unexpected error during generation: {e}")
        return 1


def main():
    """Main function with hardcoded parameters from your original command."""
    
    # Your original command parameters hardcoded
    checkpoint = "checkpoints/crossdocked_fa_cond_temp.ckpt"
    pdbfile = "src/models/diffsbdd/data/drd2_strucutres/7e2z.pdb"
    outfile = "results/diffsbdd_150x150_gen/7e2z_150x150_gen.sdf"
    ref_ligand = "src/models/diffsbdd/data/drd2_strucutres/7e2z_9sc.sdf"
    n_samples = 150
    timesteps = 500
    num_nodes_lig = 32
    ligand_chunk_size = 50  # Common default value
    top_k = 100  # Common default value for top k
    progressive_training = False
    start_timesteps = 20
    end_timesteps = 500
    progressive_training_epochs = 100  # should be same as num_outer_epochs really
    save_traj = False
    
    
    
    
    # Create a simple namespace to hold parameters
    args = SimpleNamespace(
        checkpoint=checkpoint,
        pdbfile=pdbfile,
        outfile=outfile,
        ref_ligand=ref_ligand,
        n_samples=n_samples,
        timesteps=timesteps,
        num_nodes_lig=num_nodes_lig,
        keep_patched=False,
        save_traj=save_traj
    )
    
    # Validate input files exist
    if not os.path.exists(args.checkpoint):
        print(f"✗ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.pdbfile):
        print(f"✗ PDB file not found: {args.pdbfile}")
        sys.exit(1)
        
    if not os.path.exists(args.ref_ligand):
        print(f"✗ Reference ligand file not found: {args.ref_ligand}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    outdir = os.path.dirname(args.outfile)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        print(f"✓ Created output directory: {outdir}")
    
    print("="*60)
    print("CHECKPOINT PATCHING AND LIGAND GENERATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"PDB file: {args.pdbfile}")
    print(f"Output file: {args.outfile}")
    print(f"Reference ligand: {args.ref_ligand}")
    print(f"Samples: {args.n_samples}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Ligand nodes: {args.num_nodes_lig}")
    print("="*60 + "\n")
    
    # Step 1: Patch the checkpoint with all parameters
    try:
        patched_checkpoint = patch_checkpoint(
            args.checkpoint, 
            num_nodes_lig=args.num_nodes_lig, 
            ligand_chunk_size=ligand_chunk_size, 
            top_k=top_k,
            progressive_training=progressive_training,
            start_timesteps=start_timesteps,
            end_timesteps=end_timesteps,
            progressive_training_epochs=progressive_training_epochs
        )
    except Exception as e:
        print(f"✗ Error during checkpoint patching: {e}")
        sys.exit(1)
    
    # Step 2: Run generation
    try:
        return_code = run_generation(patched_checkpoint, args)
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return_code = 1
    
    # Step 3: Cleanup patched checkpoint if requested
    if not args.keep_patched and patched_checkpoint != args.checkpoint:
        try:
            os.remove(patched_checkpoint)
            print(f"✓ Cleaned up patched checkpoint: {patched_checkpoint}")
        except Exception as e:
            print(f"⚠ Could not remove patched checkpoint: {e}")
    
    if return_code == 0:
        print(f"\n🎉 All done! Check your results at: {args.outfile}")
    else:
        print(f"💥 Process failed with return code: {return_code}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())