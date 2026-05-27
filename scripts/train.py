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
from pytorch_lightning.strategies import DDPStrategy

# Import our new, clean components from the src library
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule
from src.prism.ppo_tuner.lightning_module import PPOFineTuner
from src.prism.utils import dict_to_namespace


class PTModelCheckpoint(ModelCheckpoint):
    """
    Custom Checkpoint class that saves a matching .pt file
    every time a .ckpt file is saved (and deletes it when the .ckpt is deleted).
    Supports both DiffSBDD (saves ddpm_model state_dict) and TargetDiff
    (saves the inner ScorePosNet3D state_dict via policy._model).
    """

    # ------------------------------------------------------------------
    # Diagnostic hooks — print key internal state at every epoch end so
    # we can see exactly why top-k / last.ckpt saves do or don't fire.
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            cb_metrics = trainer.callback_metrics
            monitored = cb_metrics.get(self.monitor, 'NOT FOUND')
            print(
                f"[CKPT-DBG epoch={trainer.current_epoch}] "
                f"global_step={trainer.global_step} | "
                f"_last_global_step_saved={self._last_global_step_saved} | "
                f"monitor='{self.monitor}' value={monitored} | "
                f"kth_value={self.kth_value} | "
                f"best_k_models={list(self.best_k_models.values())} | "
                f"skip={self._should_skip_saving_checkpoint(trainer)}"
            )
        super().on_train_epoch_end(trainer, pl_module)

    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        current = monitor_candidates.get(self.monitor)
        if trainer.is_global_zero:
            print(
                f"[CKPT-DBG _save_topk epoch={trainer.current_epoch}] "
                f"current={current} | kth_value={self.kth_value} | "
                f"best_k_models count={len(self.best_k_models)}"
            )
        super()._save_topk_checkpoint(trainer, monitor_candidates)

    def _save_last_checkpoint(self, trainer, monitor_candidates):
        if trainer.is_global_zero:
            print(
                f"[CKPT-DBG _save_last epoch={trainer.current_epoch}] "
                f"_save_last_checkpoint CALLED | "
                f"last_model_path={self.last_model_path}"
            )
        super()._save_last_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, trainer, filepath):
        if trainer.is_global_zero:
            print(f"[CKPT-DBG _save_checkpoint epoch={trainer.current_epoch}] writing {filepath}")
        super()._save_checkpoint(trainer, filepath)

        if trainer.is_global_zero:
            pt_path = str(filepath).replace('.ckpt', '.pt')
            lm = trainer.lightning_module

            if hasattr(lm, 'ddpm_model') and lm.ddpm_model is not None:
                # DiffSBDD path
                torch.save(lm.ddpm_model.state_dict(), pt_path)
                print(f"[PT Checkpoint] Saved DiffSBDD .pt to {pt_path}")
            elif hasattr(lm, 'policy') and hasattr(lm.policy, '_model'):
                # TargetDiff path – save in the native TargetDiff format so the
                # checkpoint can be reloaded by load_targetdiff_policy()
                torch.save({'model': lm.policy._model.state_dict()}, pt_path)
                print(f"[PT Checkpoint] Saved TargetDiff .pt to {pt_path}")

    def _remove_checkpoint(self, trainer, filepath):
        super()._remove_checkpoint(trainer, filepath)

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
        # seeds everything including numpy and torch
        # workers=True seeds the dataloader
        print(f"[SEED] Set random seed to {args.seed}")
        
    if args.datadir is not None:
        config.datadir = args.datadir
        print(f"[DATADIR] Set datadir to {args.datadir}")

    if args.logdir is not None:
        config.logdir = args.logdir
        print(f"[LOGDIR] Override logdir to {args.logdir}")
        
    if args.hotspot_path:
        # Since config is a Namespace, use dot notation:
        config.reward_params.reward_paths.feature_density = args.hotspot_path
        print(f"[HOTSPOT] Overriding hotspot path to: {args.hotspot_path}")
        
    
    if hasattr(config, 'eval_params'):
        # We assume the file is always named 'train_smiles.npy' and lives in the datadir
        smiles_path = Path(config.datadir) / 'train_smiles.npy'
        
        # Overwrite the hardcoded path from YAML
        config.eval_params.smiles_file = str(smiles_path)
        print(f"[EVAL] Auto-updated smiles_file path to: {smiles_path}")
    
    # --- 2. Instantiate the DataModule ---
    datamodule = LigandPocketDataModule(config)

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
    
    # --- 3. Instantiate the LightningModule ---
    # node_histogram is only needed by DiffSBDD; skip for TargetDiff.
    model_type = getattr(config, 'model_type', 'diffsbdd')
    if model_type == 'targetdiff':
        node_histogram = None
    else:
        histogram_file = Path(config.datadir, 'size_distribution.npy')
        if not histogram_file.exists():
            raise FileNotFoundError(f"Histogram file not found at {histogram_file}")
        node_histogram = np.load(histogram_file).tolist()

    model = PPOFineTuner(
        config=config,
        warm_start_checkpoint=args.warm_start_from_ddpm,
        node_histogram=node_histogram,
        checkpoint_dir=checkpoint_dir)
    

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

    num_gpus = config.gpus if isinstance(config.gpus, int) else len(config.gpus)
    strategy = DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else 'auto'

    trainer = pl.Trainer(
        max_epochs=config.ppo.num_outer_epochs,
        accelerator='gpu',
        devices=config.gpus,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        enable_progress_bar=config.enable_progress_bar,
        num_sanity_val_steps=config.num_sanity_val_steps,
        logger=wandb_logger,
        limit_train_batches=1,
    )

    # --- 5. Start Training ---
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint, weights_only=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to your config.yaml")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to resume PPO training from.")
    parser.add_argument('--warm_start_from_ddpm', type=str, default=None, help="Path to pretrained DDPM checkpoint for warm start")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--datadir', type=str, default=None, help="Path to the dataset")
    parser.add_argument('--hotspot_path', type=str, default=None, help="Override path for FeatureDensityReward hotspot pkl")
    
    # [FIX 4] Added missing arguments to avoid crash
    parser.add_argument('--logdir', type=str, default=None, help="Override log directory (Safe Scratch)")
    parser.add_argument('--dataset_name', type=str, default=None, help="Explicit dataset name (fixes scratch naming bug)")
    
    args = parser.parse_args()
    main(args)