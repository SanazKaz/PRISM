"""
Optional gradient / weight-update diagnostics for PPOFineTuner.

Logs, once per (throttled) epoch, the post-accumulation, post-clip gradients
that actually update the weights, plus the per-layer weight change across the
epoch. Built to diagnose vanishing gradients or weights that never move under
PPO's manual-optimisation loop.

Why hooks (not wandb.watch): under manual optimisation with gradient
accumulation, the only place the *real* update gradients are live is inside
on_before_optimizer_step (right after clip_grad_norm_, before
optimizer.zero_grad(set_to_none=True) in ppo_algorithm.train_step).

Off by default (config.grad_logging.enabled); toggle per-run. This class owns
all the state and logic; PPOFineTuner keeps only one-line delegating calls
from the Lightning hooks it needs (on_fit_start, on_train_epoch_start,
on_before_optimizer_step, on_train_epoch_end) — those have to stay on the
LightningModule itself since Lightning calls hooks by name on it, not on an
arbitrary helper object.

Each hook method here takes the owning PPOFineTuner (`module`) and reads
whatever it needs off it (module.policy, module.trainer, module.current_epoch,
module.logger) rather than the caller unpacking those into separate args.
"""

import math

import torch

try:
    import wandb
except ImportError:
    wandb = None


class GradDiagnostics:
    def __init__(self, config):
        gl = getattr(config, 'grad_logging', None)
        self.enabled           = bool(getattr(gl, 'enabled', False))              if gl is not None else False
        self.every_n_epochs    = int(getattr(gl, 'every_n_epochs', 1))            if gl is not None else 1
        self.log_histogram     = bool(getattr(gl, 'log_histogram', True))         if gl is not None else True
        self.per_block_hist    = bool(getattr(gl, 'per_block_histogram', True))   if gl is not None else True
        self.max_hist_elements = int(getattr(gl, 'max_hist_elements', 1_000_000)) if gl is not None else 1_000_000

        # Per-epoch transient state (reset each logging epoch).
        self._w_start = None
        self._grad_flat = []            # list of (block_key, 1-D cpu tensor)
        self._grad_block_sqsum = {}     # block_key -> sum of grad^2
        self._grad_global_sqsum = 0.0
        self._grad_captured = False
        self._hook_logged = False       # one-time "hook fired" confirmation print

        if self.enabled:
            print(f"[GradLog] Enabled (every_n_epochs={self.every_n_epochs}, "
                  f"histogram={self.log_histogram}, per_block={self.per_block_hist}).")

    @staticmethod
    def _block_key(name: str) -> str:
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

    def is_logging_epoch(self, module) -> bool:
        """True only on rank 0, when enabled, on a throttled logging epoch."""
        return (
            self.enabled
            and module.trainer is not None
            and module.trainer.is_global_zero
            and ((module.current_epoch + 1) % self.every_n_epochs == 0)
        )

    def on_fit_start(self, module):
        """Pin all grad/* series to an 'epoch' x-axis so per-epoch histograms
        render cleanly and never move the wandb step cursor backward."""
        if self.enabled and module.trainer is not None and module.trainer.is_global_zero:
            exp = getattr(module.logger, 'experiment', None)
            if exp is not None and hasattr(exp, 'define_metric'):
                exp.define_metric('grad/*', step_metric='epoch')

    def on_train_epoch_start(self, module):
        """Snapshot trainable weights so we can measure the per-epoch update."""
        if not self.is_logging_epoch(module):
            return
        self._w_start = {
            name: p.detach().clone()
            for name, p in module.policy.named_parameters()
            if p.requires_grad
        }
        self._grad_flat = []
        self._grad_block_sqsum = {}
        self._grad_global_sqsum = 0.0
        self._grad_captured = False

    def on_before_optimizer_step(self, module):
        """Capture the live, post-clip gradients of the first optimizer step
        of the epoch. Fires under manual optimisation because the code calls
        optimizer.step() on the LightningOptimizer wrapper."""
        if not self.is_logging_epoch(module):
            return
        if not self._hook_logged:
            print("[GradLog] on_before_optimizer_step fired — capturing live gradients.")
            self._hook_logged = True
        if self._grad_captured:
            return  # one representative snapshot per epoch is enough

        remaining = self.max_hist_elements
        for name, p in module.policy.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.detach()
            sq = torch.sum(g * g).item()
            self._grad_global_sqsum += sq
            key = self._block_key(name)
            self._grad_block_sqsum[key] = self._grad_block_sqsum.get(key, 0.0) + sq
            if self.log_histogram and remaining > 0:
                flat = g.flatten()
                if flat.numel() > remaining:
                    flat = flat[:remaining]
                self._grad_flat.append((key, flat.float().cpu()))
                remaining -= flat.numel()
        self._grad_captured = True

    def log_diagnostics(self, module):
        """Emit one consolidated wandb log per epoch: grad norms, grad
        histograms, and weight-update magnitudes. Rank-0 / logging-epoch only."""
        if not self.is_logging_epoch(module):
            return
        exp = getattr(module.logger, 'experiment', None)
        if exp is None:
            return

        metrics = {'epoch': module.current_epoch}

        # --- gradient norms (post-accumulation, post-clip) ---
        if self._grad_captured:
            metrics['grad/grad_norm_global'] = math.sqrt(self._grad_global_sqsum)
            for key, sq in self._grad_block_sqsum.items():
                metrics[f'grad/grad_norm/{key}'] = math.sqrt(sq)
        else:
            print("[GradLog] WARNING: no gradients captured this epoch "
                  "(on_before_optimizer_step may not have fired). See plan fallback.")

        # --- gradient histograms ---
        if self.log_histogram and wandb is not None and self._grad_flat:
            all_grads = torch.cat([t for _, t in self._grad_flat])
            metrics['grad/hist/all'] = wandb.Histogram(all_grads.numpy())
            if self.per_block_hist:
                by_block = {}
                for key, t in self._grad_flat:
                    by_block.setdefault(key, []).append(t)
                for key, tensors in by_block.items():
                    metrics[f'grad/hist/{key}'] = wandb.Histogram(torch.cat(tensors).numpy())

        # --- weight-update magnitude (||w_end - w_start|| per block) ---
        if self._w_start is not None:
            total_delta_sq = 0.0
            block_delta_sq, block_w_sq = {}, {}
            for name, p in module.policy.named_parameters():
                if not p.requires_grad or name not in self._w_start:
                    continue
                delta_sq = torch.sum((p.detach() - self._w_start[name]) ** 2).item()
                w_sq = torch.sum(p.detach() ** 2).item()
                total_delta_sq += delta_sq
                key = self._block_key(name)
                block_delta_sq[key] = block_delta_sq.get(key, 0.0) + delta_sq
                block_w_sq[key] = block_w_sq.get(key, 0.0) + w_sq
            metrics['grad/weight_update_l2_global'] = math.sqrt(total_delta_sq)
            for key in block_delta_sq:
                d = math.sqrt(block_delta_sq[key])
                metrics[f'grad/weight_update_l2/{key}'] = d
                metrics[f'grad/weight_update_rel/{key}'] = d / (math.sqrt(block_w_sq[key]) + 1e-12)

        exp.log(metrics, step=module.trainer.global_step)

        # Reset transient state for the next logging epoch.
        self._w_start = None
        self._grad_flat = []
        self._grad_block_sqsum = {}
        self._grad_global_sqsum = 0.0
        self._grad_captured = False
