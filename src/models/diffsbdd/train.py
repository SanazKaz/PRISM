import argparse
from argparse import Namespace
from pathlib import Path
import warnings
from pytorch_lightning.callbacks import Callback
from datetime import timedelta
import torch
import pytorch_lightning as pl
import yaml
import numpy as np
from lightning_modules import LigandPocketDDPM
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
import os
import time

def set_seed(seed):
    """Set all random seeds for reproducibility in PPO training"""
    seed_everything(seed, workers=True) 
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    print(f"Set global random seed to {seed}")

def merge_args_and_yaml(args, config_dict):
    """Merge command line arguments with YAML configuration"""
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} from the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value
    return args

# REMOVED: merge_configs() - no longer needed since we only load DDPM weights

def create_ppo_ready_checkpoint(pl_module, save_path, original_checkpoint_info=None):
    """
    Create a PPO-ready checkpoint that's compatible with current architecture.
    This avoids all backward compatibility issues.
    
    Args:
        pl_module: The initialized LigandPocketDDPM module
        save_path: Path where to save the new checkpoint
        original_checkpoint_info: Optional info about source checkpoint
    """
    print(f" Creating PPO-ready checkpoint...")
    
    # 🔧 SIMPLIFIED: Create checkpoint manually (no trainer needed)
    
    # Create checkpoint data with flexible metadata
    checkpoint_data = {
        'state_dict': pl_module.state_dict(),
        'hyper_parameters': pl_module.hparams,
        'epoch': 0,  # Starting fresh
        'global_step': 0,  # Starting fresh
        'pytorch-lightning_version': pl.__version__,
        'state_dict_key': 'state_dict',
        'lr_schedulers': [],
        'optimizer_states': [],
        #  Flexible metadata that won't break loading
        'model_info': {
            'architecture_version': 'ppo_compatible',
            'ddpm_pretrained': original_checkpoint_info is not None,
            'has_ppo_components': hasattr(pl_module.hparams, 'ppo_params') and pl_module.hparams.ppo_params is not None,
            'creation_timestamp': time.time(),
        }
    }
    
    # Save the checkpoint
    torch.save(checkpoint_data, save_path)
    
    print(f"PPO-ready checkpoint saved to: {save_path}")
    if original_checkpoint_info:
        print(f"  Based on: {original_checkpoint_info}")
    print(f"  Architecture: Current PPO-compatible version")
    print(f"  Training state: Reset to epoch 0, step 0")
    
    return save_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--create_ppo_checkpoint', action='store_true', 
                   help='Create PPO-ready checkpoint and exit (useful for migration)')
    args = p.parse_args()

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    ckpt_path = None if args.resume is None else Path(args.resume)

    args = merge_args_and_yaml(args, config)
    print(f"FINAL PPO PARAMS AFTER MERGE: {args.ppo_params.__dict__}")
    print(f"DIRECTORY OF LOGS: {args.logdir}")

    out_dir = Path(args.logdir, args.run_identifier)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    
    run_identifier = f"{args.run_identifier}_seed_{args.seed}"
    print(f"RUN IDENTIFIER: {run_identifier}")


    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        ppo_params=args.ppo_params,
        run_identifier=run_identifier
    )
    
    # Load checkpoint weights selectively and create PPO-ready version
    original_checkpoint_info = None
    if ckpt_path is not None:
        print(f" Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ddpm_state = {
            k.replace("ddpm.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("ddpm.")
        }
        pl_module.ddpm.load_state_dict(ddpm_state, strict=True)
        print("Warm-started DDPM weights – skipping optimizer restoration")
        original_checkpoint_info = str(ckpt_path)
        
        # CREATE PPO-READY CHECKPOINT AUTOMATICALLY
        ppo_checkpoint_dir = Path(args.logdir, 'ppo_checkpoints')
        ppo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ppo_checkpoint_path = ppo_checkpoint_dir / f"{args.run_name}_ppo_ready.ckpt"
        
        create_ppo_ready_checkpoint(
            pl_module, 
            ppo_checkpoint_path, 
            original_checkpoint_info
        )
        
        #  If user just wants to create checkpoint and exit
        if args.create_ppo_checkpoint:
            print(f" PPO checkpoint created successfully! Use this for future training:")
            print(f"   --resume {ppo_checkpoint_path}")
            exit(0)

    # Continue with normal training setup
    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='Crossdock_QED',
        group=getattr(args.wandb_params, 'group', None),
        name=f"{args.run_name}_seed_{args.seed}", 
        id=f"{args.run_name}_seed_{args.seed}",
        resume='allow' if args.resume is not None else None,
        entity=getattr(args.wandb_params, 'entity', 'sanazkazeminia97'),
        mode=getattr(args.wandb_params, 'mode', 'online'),
        log_model=False,    
    )
    
    # Ensure checkpoint directory is absolute and tied to logdir
    checkpoint_dir = Path(args.logdir, args.run_name, 'checkpoints')  # Add args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train/reward_mean_epoch',
        mode='max',  # Maximize reward
        save_top_k=1,
        dirpath=str(checkpoint_dir),
        filename='ppo-{epoch:02d}-{train_reward_mean_epoch:.2f}',
        save_last=True,  # Save the last model for easy resumption
        every_n_epochs=1  # Save every epoch (outer loop)
    )
    
        
    class LimitValidationStepCallback(Callback):
        def on_validation_end(self, trainer, pl_module):
            # Reset the should_stop flag
            trainer.should_stop = False
            # Force the training to continue
            trainer._should_stop_early = False
            
        def on_train_epoch_end(self, trainer, pl_module):
            # Ensure we continue training after each epoch
            trainer.should_stop = False
            trainer._should_stop_early = False   
    
    trainer = pl.Trainer(
        max_epochs=args.ppo_params.num_outer_epochs,  # Use args.ppo_params
        logger=logger,
        limit_train_batches=1,
        callbacks=[
                   checkpoint_callback, 
                   LimitValidationStepCallback(),
                   LearningRateMonitor(logging_interval="step")],
        enable_progress_bar=args.enable_progress_bar,  # Default to False for PPO
        num_sanity_val_steps=args.num_sanity_val_steps,  # Disable by default
        accelerator='gpu',
        devices=args.gpus,
        strategy = DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        log_every_n_steps=1,  # Log every step for fine-grained tracking
    )
    

    trainer.fit(model=pl_module)