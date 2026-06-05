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
            from src.prism.models.targetdiff_inference import make_targetdiff_reconstruction_fn
            reconstruction_fn = make_targetdiff_reconstruction_fn()
            # Frozen reference policy — never updated, anchors KL penalty to pretrained prior
            if getattr(self.config.ppo, 'ref_kl_coef', 0.0) > 0.0:
                self.ref_policy, _ = build_targetdiff_policy(
                    config=self.config,
                    device=device,
                    warm_start_checkpoint=warm_start_checkpoint,
                )
                self.ref_policy.eval()
                for p in self.ref_policy.parameters():
                    p.requires_grad_(False)
                print("[Init] Frozen reference policy loaded for KL anchor.")
            else:
                self.ref_policy = None
        else:
            self.policy, self.ddpm_model, self.dataset_info = build_diffsbdd_policy(
                config=self.config,
                device=device,
                node_histogram=node_histogram,
                warm_start_checkpoint=warm_start_checkpoint,
            )
            reconstruction_fn = None  # DiffSBDD uses build_molecule (default)

        # Attach the data directory so reward functions can resolve relative paths.
        self.dataset_info['datadir'] = self.config.datadir

        self.reward_manager = get_reward_manager(
            config=self.config,
            dataset_info=self.dataset_info,
            ddpm_module=self.policy,  # policy satisfies the virtual-node interface
            reconstruction_fn=reconstruction_fn,
        )

        self.ppo_algorithm = PPOAlgorithm(
            policy_network=self.policy,
            ref_policy=getattr(self, 'ref_policy', None),
            reward_function=self.reward_manager,
            config=self.config,
            dataset_info=self.dataset_info,
            checkpoint_dir=checkpoint_dir,
        )
        self.freeze_parameters()
        self._init_grad_logging()

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
    # Gradient / weight-update diagnostics
    #
    # Logs, once per (throttled) epoch, the post-accumulation, post-clip
    # gradients that actually update the weights, plus the per-layer weight
    # change across the epoch. Built to diagnose vanishing gradients or
    # weights that never move under PPO's manual-optimisation loop.
    #
    # Why hooks (not wandb.watch): under manual optimisation with gradient
    # accumulation, the only place the *real* update gradients are live is
    # inside on_before_optimizer_step (right after clip_grad_norm_, before
    # optimizer.zero_grad(set_to_none=True) in ppo_algorithm.train_step).
    # ------------------------------------------------------------------

    def _init_grad_logging(self):
        """Read the optional `grad_logging` config block and init transient state."""
        gl = getattr(self.config, 'grad_logging', None)
        self._grad_log_enabled        = bool(getattr(gl, 'enabled', False))            if gl is not None else False
        self._grad_log_every_n        = int(getattr(gl, 'every_n_epochs', 1))          if gl is not None else 1
        self._grad_log_histogram      = bool(getattr(gl, 'log_histogram', True))       if gl is not None else True
        self._grad_log_per_block_hist = bool(getattr(gl, 'per_block_histogram', True)) if gl is not None else True
        self._grad_log_max_hist       = int(getattr(gl, 'max_hist_elements', 1_000_000)) if gl is not None else 1_000_000

        # Per-epoch transient state (reset each logging epoch).
        self._w_start = None
        self._grad_flat = []            # list of (block_key, 1-D cpu tensor)
        self._grad_block_sqsum = {}     # block_key -> sum of grad^2
        self._grad_global_sqsum = 0.0
        self._grad_captured = False
        self._grad_hook_logged = False  # one-time "hook fired" confirmation print

        if self._grad_log_enabled:
            print(f"[GradLog] Enabled (every_n_epochs={self._grad_log_every_n}, "
                  f"histogram={self._grad_log_histogram}, per_block={self._grad_log_per_block_hist}).")

    @staticmethod
    def _grad_block_key(name: str) -> str:
        """Map a trainable param name to a coarse block bucket matching freeze_except.

        TargetDiff trainable names are e.g. 'v_inference.0.weight' and
        'refine_net.base_block.6.<...>' (no '_model.' prefix — see
        TargetDiffPolicy.named_parameters).
        """
        if 'v_inference' in name:
            return 'v_inference'
        marker = 'base_block.'
        if marker in name:
            idx = name.split(marker, 1)[1].split('.', 1)[0]
            return f'base_block.{idx}'
        return 'other'

    def _is_grad_log_epoch(self) -> bool:
        """True only on rank 0, when enabled, on a throttled logging epoch."""
        return (
            self._grad_log_enabled
            and self.trainer is not None
            and self.trainer.is_global_zero
            and ((self.current_epoch + 1) % self._grad_log_every_n == 0)
        )

    def on_fit_start(self):
        """Pin all grad/* series to an 'epoch' x-axis so per-epoch histograms
        render cleanly and never move the wandb step cursor backward."""
        if self._grad_log_enabled and self.trainer is not None and self.trainer.is_global_zero:
            exp = getattr(self.logger, 'experiment', None)
            if exp is not None and hasattr(exp, 'define_metric'):
                exp.define_metric('grad/*', step_metric='epoch')

    def on_train_epoch_start(self):
        """Snapshot trainable weights so we can measure the per-epoch update."""
        if not self._is_grad_log_epoch():
            return
        self._w_start = {
            name: p.detach().clone()
            for name, p in self.policy.named_parameters()
            if p.requires_grad
        }
        # Reset per-epoch grad accumulators.
        self._grad_flat = []
        self._grad_block_sqsum = {}
        self._grad_global_sqsum = 0.0
        self._grad_captured = False

    def on_before_optimizer_step(self, optimizer):
        """Capture the live, post-clip gradients of the first optimizer step
        of the epoch. Fires under manual optimisation because the code calls
        optimizer.step() on the LightningOptimizer wrapper."""
        if not self._is_grad_log_epoch():
            return
        if not self._grad_hook_logged:
            print("[GradLog] on_before_optimizer_step fired — capturing live gradients.")
            self._grad_hook_logged = True
        if self._grad_captured:
            return  # one representative snapshot per epoch is enough

        remaining = self._grad_log_max_hist
        for name, p in self.policy.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.detach()
            sq = torch.sum(g * g).item()
            self._grad_global_sqsum += sq
            key = self._grad_block_key(name)
            self._grad_block_sqsum[key] = self._grad_block_sqsum.get(key, 0.0) + sq
            if self._grad_log_histogram and remaining > 0:
                flat = g.flatten()
                if flat.numel() > remaining:
                    flat = flat[:remaining]
                self._grad_flat.append((key, flat.float().cpu()))
                remaining -= flat.numel()
        self._grad_captured = True

    def _log_grad_diagnostics(self):
        """Emit one consolidated wandb log per epoch: grad norms, grad
        histograms, and weight-update magnitudes. Rank-0 / logging-epoch only."""
        if not self._is_grad_log_epoch():
            return
        exp = getattr(self.logger, 'experiment', None)
        if exp is None:
            return

        import math
        try:
            import wandb
        except ImportError:
            wandb = None

        metrics = {'epoch': self.current_epoch}

        # --- gradient norms (post-accumulation, post-clip) ---
        if self._grad_captured:
            metrics['grad/grad_norm_global'] = math.sqrt(self._grad_global_sqsum)
            for key, sq in self._grad_block_sqsum.items():
                metrics[f'grad/grad_norm/{key}'] = math.sqrt(sq)
        else:
            print("[GradLog] WARNING: no gradients captured this epoch "
                  "(on_before_optimizer_step may not have fired). See plan fallback.")

        # --- gradient histograms ---
        if self._grad_log_histogram and wandb is not None and self._grad_flat:
            all_grads = torch.cat([t for _, t in self._grad_flat])
            metrics['grad/hist/all'] = wandb.Histogram(all_grads.numpy())
            if self._grad_log_per_block_hist:
                by_block = {}
                for key, t in self._grad_flat:
                    by_block.setdefault(key, []).append(t)
                for key, tensors in by_block.items():
                    metrics[f'grad/hist/{key}'] = wandb.Histogram(torch.cat(tensors).numpy())

        # --- weight-update magnitude (||w_end - w_start|| per block) ---
        if self._w_start is not None:
            total_delta_sq = 0.0
            block_delta_sq, block_w_sq = {}, {}
            for name, p in self.policy.named_parameters():
                if not p.requires_grad or name not in self._w_start:
                    continue
                delta_sq = torch.sum((p.detach() - self._w_start[name]) ** 2).item()
                w_sq = torch.sum(p.detach() ** 2).item()
                total_delta_sq += delta_sq
                key = self._grad_block_key(name)
                block_delta_sq[key] = block_delta_sq.get(key, 0.0) + delta_sq
                block_w_sq[key] = block_w_sq.get(key, 0.0) + w_sq
            metrics['grad/weight_update_l2_global'] = math.sqrt(total_delta_sq)
            for key in block_delta_sq:
                d = math.sqrt(block_delta_sq[key])
                metrics[f'grad/weight_update_l2/{key}'] = d
                metrics[f'grad/weight_update_rel/{key}'] = d / (math.sqrt(block_w_sq[key]) + 1e-12)

        exp.log(metrics, step=self.trainer.global_step)

        # Reset transient state for the next logging epoch.
        self._w_start = None
        self._grad_flat = []
        self._grad_block_sqsum = {}
        self._grad_global_sqsum = 0.0
        self._grad_captured = False

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
        logs = self.ppo_algorithm.train_step(
            pocket_batch=batch,
            current_epoch=self.current_epoch,
            optimizer=opt,
            backward_fn=self.manual_backward,
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
        """Emit gradient/weight diagnostics, then trigger periodic validation."""
        self._log_grad_diagnostics()

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
