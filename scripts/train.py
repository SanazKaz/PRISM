# scripts/train.py

import sys
import os
from pathlib import Path

# later add in wandb logging
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add diffsbdd directory to path so its old imports work unchanged
diffsbdd_path = Path(project_root) / "src" / "models" / "diffsbdd"
sys.path.insert(1, str(diffsbdd_path))
# later replace with toml


import argparse
from argparse import Namespace
from pathlib import Path
import yaml
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

# Import our new, clean components from the src library
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule
from src.prism.ppo_tuner.lightning_module import PPOFineTuner

def dict_to_namespace(d):
    """ Recursively converts a dictionary to a namespace. """
    namespace = Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


class PTModelCheckpoint(ModelCheckpoint):
    """
    Custom Checkpoint class that saves a matching .pt file 
    every time a .ckpt file is saved (and deletes it when the .ckpt is deleted).
    """
    def _save_checkpoint(self, trainer, filepath):
        # 1. Save the standard .ckpt file
        super()._save_checkpoint(trainer, filepath)
        
        # 2. Save the matching .pt file
        if trainer.is_global_zero:
            # Swap extension to .pt
            pt_path = str(filepath).replace('.ckpt', '.pt')
            
            # Save the inner model only
            # We access the trainer's model directly
            if hasattr(trainer.lightning_module, 'ddpm_model'):
                inner_model = trainer.lightning_module.ddpm_model
                torch.save(inner_model.state_dict(), pt_path)
                print(f"[PT Checkpoint] Saved matching .pt to {pt_path}")

    def _remove_checkpoint(self, trainer, filepath):
        # 1. Delete the standard .ckpt file
        super()._remove_checkpoint(trainer, filepath)
        
        # 2. Delete the matching .pt file to keep folder clean
        if trainer.is_global_zero:
            pt_path = str(filepath).replace('.ckpt', '.pt')
            if os.path.exists(pt_path):
                os.remove(pt_path)
                print(f"[PT Checkpoint] Removed old .pt file {pt_path}")

def main(args):
    # --- 1. Load Configuration ---
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        print(f"[SEED] Set random seed to {args.seed}")
        
    if args.datadir is not None:
        config.datadir = args.datadir
        print(f"[DATADIR] Set datadir to {args.datadir}")

    if args.logdir is not None:
        config.logdir = args.logdir
        print(f"[LOGDIR] Override logdir to {args.logdir}")
        
    
    if hasattr(config, 'eval_params'):
        # We assume the file is always named 'train_smiles.npy' and lives in the datadir
        smiles_path = Path(config.datadir) / 'train_smiles.npy'
        
        # Overwrite the hardcoded path from YAML
        config.eval_params.smiles_file = str(smiles_path)
        print(f"[EVAL] Auto-updated smiles_file path to: {smiles_path}")
    
    # --- 2. Instantiate the DataModule ---
    datamodule = LigandPocketDataModule(config)

    # --- 3. Instantiate the LightningModule ---
    histogram_file = Path(config.datadir, 'size_distribution.npy')
    if not histogram_file.exists():
        raise FileNotFoundError(f"Histogram file not found at {histogram_file}")
    node_histogram = np.load(histogram_file).tolist()
    
    model = PPOFineTuner(config=config, warm_start_checkpoint=args.warm_start_from_ddpm, node_histogram=node_histogram)
    
    # --- 4. Setup Callbacks and Trainer ---
    
    # If passed explicitly, use it. Otherwise try to derive it (risky on scratch).
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        # Fallback: risky if folder structure changes!
        dataset_name = Path(config.datadir).parent.name 

    checkpoint_dir = Path(config.logdir, 
                          config.run_identifier, 
                          'checkpoints',
                          dataset_name,
                          f'seed={args.seed}',
                          )
    
    print(f"[********* CHECKPOINT DIR **********] {checkpoint_dir}")

    checkpoint_callback = PTModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor='train/reward_mean', 
        mode='max',
        save_last=True,
        filename="epoch={epoch:02d}-reward={train/reward_mean:.2f}",
        save_top_k=3,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False 
    )
    
    run_name = f"{config.run_identifier}_{dataset_name}_seed{args.seed}"

    wandb_logger = WandbLogger(
        entity=getattr(config.wandb_params, 'entity', None),
        project=getattr(config.wandb_params, 'project', 'PRISM-Training'),
        name=run_name,
        config=config_dict,
    )

    trainer = pl.Trainer(
        max_epochs=config.ppo_params.num_outer_epochs,
        accelerator='gpu',
        devices=config.gpus,
        callbacks=[checkpoint_callback],
        enable_progress_bar=config.enable_progress_bar,
        num_sanity_val_steps=config.num_sanity_val_steps,
        logger=wandb_logger,
        limit_train_batches=1, 
    )

    # --- 5. Start Training ---
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to your config.yaml")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to resume PPO training from.")
    parser.add_argument('--warm_start_from_ddpm', type=str, default=None, help="Path to pretrained DDPM checkpoint for warm start")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--datadir', type=str, default=None, help="Path to the dataset")
    
    # [FIX 4] Added missing arguments to avoid crash
    parser.add_argument('--logdir', type=str, default=None, help="Override log directory (Safe Scratch)")
    parser.add_argument('--dataset_name', type=str, default=None, help="Explicit dataset name (fixes scratch naming bug)")
    
    args = parser.parse_args()
    main(args)