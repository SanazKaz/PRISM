# scripts/train.py

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add diffsbdd directory to path so its old imports work unchanged
diffsbdd_path = Path(project_root) / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(diffsbdd_path))
# later replace with toml


import argparse
from argparse import Namespace
from pathlib import Path
import yaml
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import our new, clean components from the src library
from src.prism.data.datamodule import LigandPocketDataModule
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

def main(args):
    # --- 1. Load Configuration ---
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    
    # --- 2. Instantiate the DataModule ---
    # The DataModule handles all data-related setup.
    datamodule = LigandPocketDataModule(config)

    # --- 3. Instantiate the LightningModule ---
    # The LightningModule handles the model and training logic.
    model = PPOFineTuner(config=config, warm_start_checkpoint=args.warm_start_from_ddpm)

    
    # --- 4. Setup Callbacks and Trainer ---
    checkpoint_dir = Path(config.logdir, config.run_identifier, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor='train/reward_mean', # Make sure your logs match this key
        mode='max',
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.ppo_params.num_outer_epochs,
        accelerator='gpu',
        devices=config.gpus,
        callbacks=[checkpoint_callback],
        enable_progress_bar=config.enable_progress_bar,
        num_sanity_val_steps=config.num_sanity_val_steps,
        # Add any other trainer flags you need from your config
    )

    # --- 5. Start Training ---
    # The trainer now gets both the model and the datamodule.
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to your config.yaml")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to resume PPO training from.")
    parser.add_argument('--warm_start_from_ddpm', type=str, default=None, help="Path to pretrained DDPM checkpoint for warm start")

    
    args = parser.parse_args()
    main(args)