# src/prism/ppo_tuner/lightning_module.py
"""
Lightning wrapper around the core PPOAlgorithm.

Responsibilities of this file
------------------------------
- Instantiate the policy, reward manager, and PPO algorithm (delegating
  model construction entirely to src/prism/models/policy_factory.py).
- Implement the Lightning training/validation hooks that tie everything
  into the Trainer loop.
- Apply parameter freezing and configure the Adam optimiser.

Everything about *how* a model is built (architecture constants, checkpoint
loading, dataset_info construction) lives in policy_factory.py, not here.
"""

import os
import pytorch_lightning as pl
import torch

from src.prism.ppo_tuner.ppo_algorithm import PPOAlgorithm
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule
from src.prism.reward.factory import get_reward_manager
from src.prism.utils import build_molecules_from_batch
from src.prism.models.policy_factory import build_diffsbdd_policy, build_targetdiff_policy
from val_analysis.smina_docking import SminaDocking
from val_analysis.metrics import MoleculeMetrics


class PPOFineTuner(pl.LightningModule):
    """
    Main LightningModule for PPO fine-tuning.
    Acts as a lightweight wrapper around the core PPOAlgorithm.
    """

    def __init__(self, config, node_histogram=None, warm_start_checkpoint=None, checkpoint_dir=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.automatic_optimization = False  # PPO manages its own optimiser steps

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}' if self.config.gpus > 0 else 'cpu')
        print(f"[INIT] PPOFineTuner | LOCAL_RANK={local_rank} | device={device}")
        model_type = getattr(self.config, 'model_type', 'diffsbdd')

        if model_type == 'targetdiff':
            self.policy, self.dataset_info = build_targetdiff_policy(
                config=self.config,
                device=device,
                warm_start_checkpoint=warm_start_checkpoint,
            )
            self.ddpm_model = None  # no LigandPocketDDPM when using TargetDiff
        else:
            self.policy, self.ddpm_model, self.dataset_info = build_diffsbdd_policy(
                config=self.config,
                device=device,
                node_histogram=node_histogram,
                warm_start_checkpoint=warm_start_checkpoint,
            )

        # Attach the data directory so reward functions can resolve relative paths.
        self.dataset_info['datadir'] = self.config.datadir

        self.reward_manager = get_reward_manager(
            config=self.config,
            dataset_info=self.dataset_info,
            ddpm_module=self.policy,  # policy satisfies the virtual-node interface
        )

        self.ppo_algorithm = PPOAlgorithm(
            policy_network=self.policy,
            reward_function=self.reward_manager,
            config=self.config,
            dataset_info=self.dataset_info,
            checkpoint_dir=checkpoint_dir,
        )
        self.freeze_parameters()

    # ------------------------------------------------------------------
    # Parameter freezing
    # ------------------------------------------------------------------

    def freeze_parameters(self):
        """Freeze all policy parameters except those whose name contains
        a substring listed in config.freeze_except.

        This design lets users control the trainable portion by editing a
        single list in the YAML config without touching code.
        """
        print("[Init] Applying freezing strategy...")
        frozen_count = 0
        unfrozen_count = 0

        trainable_layers = self.config.freeze_except
        print(f"[Init] Trainable blocks: {trainable_layers}")

        for name, param in self.policy.named_parameters():
            if any(x in name for x in trainable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1

        for param in self.policy.parameters():
            if param.requires_grad:
                unfrozen_count += 1

        print(f"[Init] {frozen_count} params frozen, {unfrozen_count} trainable.")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Adam optimiser over the unfrozen policy parameters."""
        params_to_optimize = filter(lambda p: p.requires_grad, self.policy.parameters())
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.config.ppo.lr,
            eps=1e-8,
            weight_decay=1.0e-12,
            betas=(0.9, 0.999),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        """Delegates one full PPO outer step (collect → advantage → update) to PPOAlgorithm."""
        opt = self.optimizers()
        import torch.distributed as _dist
        rank = self.trainer.global_rank
        print(f"[RANK {rank}] training_step | dist.is_initialized={_dist.is_initialized()} | "
              f"world_size={self.trainer.world_size} | device={self.device} | epoch={self.current_epoch}")

        logs = self.ppo_algorithm.train_step(
            pocket_batch=batch,
            current_epoch=self.current_epoch,
            optimizer=opt,
            backward_fn=self.manual_backward,
        )

        # Diagnostic: print what reward value is being handed to PL's metric system.
        # Compare this against [CKPT-DBG] prints to see if callback_metrics matches.
        if rank == 0:
            reward_val = logs.get('train/reward_mean', 'MISSING')
            print(
                f"[STEP-DBG epoch={self.current_epoch}] "
                f"global_step={self.trainer.global_step} | "
                f"reward_mean logged to PL={reward_val} | "
                f"keys_in_logs={list(logs.keys())}"
            )

        # sync_dist=True all-reduces each metric across DDP ranks before PL logs it,
        # so both WandB and ModelCheckpoint see the global mean (not rank 0 only).
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return logs

    def validation_step(self, batch, batch_idx):
        if self.ddpm_model is not None:
            return self.ddpm_model.validation_step(batch, batch_idx)
        # TargetDiff: full validation is handled in _run_validation (on_train_epoch_end).
        return {}

    def on_train_epoch_end(self):
        """Trigger periodic validation at intervals set by config.eval_epochs."""
        # Diagnostic: show what PL put in callback_metrics after the epoch.
        # This is exactly what ModelCheckpoint reads when deciding to save.
        if self.trainer.is_global_zero:
            cb = self.trainer.callback_metrics
            reward_in_cb = cb.get('train/reward_mean', 'NOT IN callback_metrics')
            print(
                f"[EPOCH-END-DBG epoch={self.current_epoch}] "
                f"callback_metrics['train/reward_mean']={reward_in_cb} | "
                f"global_step={self.trainer.global_step} | "
                f"all_cb_keys={list(cb.keys())}"
            )

        if (self.current_epoch + 1) % self.config.eval_epochs == 0:
            if self.ddpm_model is None:
                # TargetDiff does not use DiffSBDD's validation loop.
                print(f"[Validation] Skipping DiffSBDD validation — TargetDiff mode.")
                return
            print(f"\n{'='*80}")
            print(f"Running validation at epoch {self.current_epoch + 1}")
            print(f"{'='*80}\n")
            self._run_validation()

    def _run_validation(self):
        """Generate molecules on the validation set and log quality metrics.

        Uses the same RolloutCollector as training, then evaluates the resulting
        molecules with MoleculeMetrics and SminaDocking.  Only called for the
        DiffSBDD path (TargetDiff validation is skipped — see on_train_epoch_end).
        """
        self.ddpm_model.eval()

        try:
            with torch.no_grad():
                val_dataloader = self.trainer.datamodule.val_dataloader()

                n_eval_samples = self.config.eval_params.n_eval_samples
                eval_batch_size = self.config.eval_params.eval_batch_size

                val_batch = next(iter(val_dataloader))
                actual_batch_size = min(eval_batch_size, len(val_batch['num_pocket_nodes']))

                print(f"[Validation] Generating molecules for {actual_batch_size} pockets...")

                rollout_data = self.ppo_algorithm.collector.collect(
                    pocket_batch=val_batch,
                    current_epoch=self.current_epoch,
                    get_ligand_and_pocket_fn=self.policy.get_ligand_and_pocket,
                )

                if rollout_data['rewards'].numel() == 0:
                    print("[Validation] No valid molecules generated, skipping metrics.")
                    return

                xh_lig = rollout_data['molecules'][0]
                global_lig_mask = rollout_data['masks'][0]

                print(f"[Validation] Generated {len(torch.unique(global_lig_mask))} molecules")

                molecules, mol_to_batch_idx = build_molecules_from_batch(
                    xh_lig,
                    global_lig_mask,
                    self.dataset_info,
                    self.policy,
                )

                print(f"[Validation] Successfully built {len(molecules)} valid RDKit molecules")

                names = val_batch.get('names', None)

                metrics_calculator = MoleculeMetrics(dataset_info=self.dataset_info)
                basic_metrics = metrics_calculator.evaluate_batch(molecules)

                docking_calculator = SminaDocking(dataset_info=self.dataset_info, local_opt=False, timeout=60)
                docking_metrics = docking_calculator.dock_batch(molecules, names=names)

                metrics = {**basic_metrics, **docking_metrics}

                val_metrics = {f"val/{key}": value for key, value in metrics.items()}
                self.log_dict(val_metrics, on_step=False, on_epoch=True)

                print(f"\n[Validation Results - Epoch {self.current_epoch + 1}]")
                print("-" * 60)
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                print("-" * 60 + "\n")

        except Exception as e:
            print(f"[Validation] Error during validation: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Always restore train mode regardless of success/failure.
            self.ddpm_model.train()
