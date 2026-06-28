"""
Per-step importance-ratio distribution logger (REMOVABLE monkey-patch).
=======================================================================

Purpose
-------
Characterise the PPO importance ratio  r = pi_new / pi_old  as a function of the
diffusion timestep t, across *all* molecules in the rollout (not a single one).
Motivated by GRPO-Guard (arXiv: "GRPO-Guard"), which observes that in GRPO/PPO
on flow/diffusion models the ratio distribution is

  * left-shifted   (mean < 1, pronounced at certain noise levels), and
  * heteroskedastic (variance differs substantially across timesteps),

both of which break the clip mechanism (positive-advantage samples escape the
upper clip bound -> over-optimization). This module measures that bias so you
can decide whether a correction (e.g. ratio normalisation) is warranted.

Maths
-----
For molecule i at step t the importance ratio is

    r_i(t) = exp( log pi_new(move_i | t) - log pi_old(move_i | t) ) = exp(d_i(t))

`log pi_old` is the stored sampling log-prob (`old_log_probs`); `log pi_new` is
recomputed with the *current* policy via the exact PPO path (`loss._get_log_probs`).
Theory: E_old[r] = 1 exactly (importance weight). A measured per-step mean below 1
is therefore a finite-sample / heavy-tail bias (or genuine drift) — the thing
GRPO-Guard corrects.

Per step we keep five running accumulators (over molecules, DDP ranks AND epochs):

    count, sum_d, sumsq_d, sum_r, sumsq_r, clip_count

from which:

    logratio_mean = sum_d / count
    logratio_std  = sqrt(sumsq_d/count - logratio_mean^2)
    ratio_mean    = sum_r / count
    ratio_var     = sumsq_r/count - ratio_mean^2
    clipfrac      = clip_count / count        (|r - 1| > clip_range)

This single mechanism gives molecule-averaging, rank-averaging (all-reduce the
sums) and epoch-averaging (keep adding) — all exact.

Usage
-----
    from src.prism.analysis.ratio_diagnostics import attach_ratio_logging
    handle = attach_ratio_logging(ppo_algorithm, every_n_epochs=5)
    ...
    handle.detach()     # fully restores the patched methods

It writes <checkpoint_dir>/ratio_diagnostics/{ratio_dist_epXXXX.npz, .png} plus a
cumulative ratio_dist_avg.{npz,png}, and merges summary scalars into the metrics
dict returned by train_step (so they land in WandB).

This file touches nothing in the training hot path until attached, and removing
the attach call (or unsetting the PRISM_RATIO_DIAG env var) reverts everything.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# Reuse the EXACT log-prob path PPO uses (same mask reindexing / t normalisation).
from src.prism.ppo_tuner.loss import _get_log_probs


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


class RatioDistributionLogger:
    """Monkey-patches a PPOAlgorithm to record per-step ratio statistics.

    Wraps two methods on the live algorithm instance:
      * buffer.load_rollout_data  -> stash the FULL per-step tensors before the
        band-slicing in train_step overwrites them.
      * train_step                -> after the normal update, sweep every stored
        diffusion step with the updated policy and log the ratio distribution.

    Nothing is modified inside the algorithm's own source; .detach() restores the
    original bound methods exactly.
    """

    def __init__(self, algo, every_n_epochs: int = 5, out_dir=None,
                 clip_range: float | None = None, logratio_clamp: float = 30.0,
                 accumulate: bool = True):
        self.algo = algo
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.logratio_clamp = float(logratio_clamp)
        self.accumulate = bool(accumulate)

        cfg_clip = getattr(getattr(algo, 'config', None), 'ppo', None)
        self.clip_range = float(clip_range) if clip_range is not None \
            else float(getattr(cfg_clip, 'clip_range', 0.2))

        if out_dir is not None:
            self.out_dir = Path(out_dir)
        else:
            self.out_dir = Path(getattr(algo, 'checkpoint_dir', '.')) / 'ratio_diagnostics'

        # Stash for the full (unsliced) per-step tensors, refreshed each rollout.
        self._full = {}
        # Cumulative cross-epoch accumulators (numpy float64), lazily sized.
        self._acc = None

        # Saved originals for detach().
        self._orig_train_step = None
        self._orig_load = None
        self._attached = False

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self):
        if self._attached:
            return self
        self._orig_load = self.algo.buffer.load_rollout_data
        self._orig_train_step = self.algo.train_step
        self.algo.buffer.load_rollout_data = self._wrapped_load
        self.algo.train_step = self._wrapped_train_step
        self._attached = True
        if _is_rank0():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ratio-diag] attached (every {self.every_n_epochs} epochs, "
                  f"clip_range={self.clip_range}); writing to {self.out_dir}")
        return self

    def detach(self):
        if not self._attached:
            return
        self.algo.buffer.load_rollout_data = self._orig_load
        self.algo.train_step = self._orig_train_step
        self._attached = False
        print("[ratio-diag] detached (original methods restored).")

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _wrapped_load(self, rollout_data: dict):
        # Run the real loader first, then stash clones of the FULL per-step
        # tensors (train_step later overwrites buffer.* with the sliced band).
        self._orig_load(rollout_data)
        if getattr(self.algo.buffer, 'data_loaded', False):
            self._full = {
                'latents': self.algo.buffer.latents.detach().clone(),
                'next_latents': self.algo.buffer.next_latents.detach().clone(),
                'old_log_probs': self.algo.buffer.old_log_probs.detach().clone(),
                'timesteps': self.algo.buffer.timesteps.detach().clone(),
            }
        else:
            self._full = {}

    def _wrapped_train_step(self, *args, **kwargs):
        metrics = self._orig_train_step(*args, **kwargs)
        # current_epoch is the 2nd positional arg of train_step, or a kwarg.
        epoch = kwargs.get('current_epoch', None)
        if epoch is None and len(args) >= 2:
            epoch = args[1]
        epoch = int(epoch) if epoch is not None else 0

        if self._full and (epoch % self.every_n_epochs == 0):
            try:
                summary = self._sweep_and_log(epoch)
                if isinstance(metrics, dict) and summary:
                    metrics.update(summary)
            except Exception as e:   # never let a diagnostic break training
                if _is_rank0():
                    print(f"[ratio-diag] sweep skipped (epoch {epoch}): {e}")
        return metrics

    # ------------------------------------------------------------------
    # The sweep
    # ------------------------------------------------------------------

    def _sweep_and_log(self, epoch: int) -> dict:
        algo = self.algo
        policy = algo.policy_network
        total_timesteps = algo.config.model.total_timesteps

        latents = self._full['latents']
        next_latents = self._full['next_latents']
        old_log_probs = self._full['old_log_probs']   # [M, T]
        timesteps = self._full['timesteps']           # [M, T]
        xh_lig, xh_pock = algo.buffer.molecules         # full, not sliced
        lig_mask, poc_mask = algo.buffer.masks

        # Alignment (mirror rollout_collector): old_log_probs is stored over the
        # full chain (T+1 entries), but latents/next_latents/timesteps cover the T
        # transitions (z_states[:, :-1] / [:, 1:]). The collector aligns by taking
        # old_log_probs[:, -T:]; do the same so index ti pairs the SAME transition
        # across all tensors (otherwise the ratio compares mismatched steps). The
        # min() is a defensive guard against any further off-by-one.
        device = old_log_probs.device
        M = old_log_probs.shape[0]
        T = min(latents.shape[1], next_latents.shape[1], timesteps.shape[1])
        if old_log_probs.shape[1] != T:
            old_log_probs = old_log_probs[:, -T:]

        # Per-step accumulators for THIS measurement (summed over molecules/ranks).
        count = torch.zeros(T, device=device)
        sum_d = torch.zeros(T, device=device)
        sumsq_d = torch.zeros(T, device=device)
        sum_r = torch.zeros(T, dtype=torch.float64, device=device)
        sumsq_r = torch.zeros(T, dtype=torch.float64, device=device)
        clip_count = torch.zeros(T, device=device)
        t_value = torch.zeros(T, device=device)

        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            for ti in range(T):
                timestep_batch = {
                    'molecules': (xh_lig, xh_pock),
                    'masks': (lig_mask, poc_mask),
                    'latents': latents[:, ti],
                    'next_latents': next_latents[:, ti],
                    'timestep': timesteps[:, ti],
                }
                new_lp = _get_log_probs(policy, timestep_batch, total_timesteps)
                old_lp = old_log_probs[:, ti]
                d = (new_lp - old_lp).float()                       # [m_local]
                r = torch.exp(d.clamp(-self.logratio_clamp, self.logratio_clamp)).double()

                count[ti] = d.numel()
                sum_d[ti] = d.sum()
                sumsq_d[ti] = (d * d).sum()
                sum_r[ti] = r.sum()
                sumsq_r[ti] = (r * r).sum()
                clip_count[ti] = ((r - 1.0).abs() > self.clip_range).float().sum()
                t_value[ti] = float(timesteps[0, ti].item())
        if was_training:
            policy.train()

        # DDP: sum the accumulators across ranks so stats cover all molecules.
        if dist.is_initialized():
            for tns in (count, sum_d, sumsq_d, clip_count):
                dist.all_reduce(tns, op=dist.ReduceOp.SUM)
            for tns in (sum_r, sumsq_r):
                dist.all_reduce(tns, op=dist.ReduceOp.SUM)

        cur = {
            't': t_value.detach().cpu().numpy(),
            'count': count.detach().cpu().numpy().astype(np.float64),
            'sum_d': sum_d.detach().cpu().numpy().astype(np.float64),
            'sumsq_d': sumsq_d.detach().cpu().numpy().astype(np.float64),
            'sum_r': sum_r.detach().cpu().numpy(),
            'sumsq_r': sumsq_r.detach().cpu().numpy(),
            'clip_count': clip_count.detach().cpu().numpy().astype(np.float64),
        }

        if self.accumulate:
            if self._acc is None:
                self._acc = {k: cur[k].copy() for k in
                             ('count', 'sum_d', 'sumsq_d', 'sum_r', 'sumsq_r', 'clip_count')}
                self._acc['t'] = cur['t'].copy()
            else:
                for k in ('count', 'sum_d', 'sumsq_d', 'sum_r', 'sumsq_r', 'clip_count'):
                    self._acc[k] += cur[k]

        # Derive on ALL ranks (sums were already all-reduced, so every rank gets
        # identical stats). Only rank 0 writes files. Returning the same summary
        # dict on every rank keeps log_dict(sync_dist=True) collectives balanced
        # — a rank-0-only return can hang DDP at the epoch boundary.
        stats = self._derive(cur)
        if _is_rank0():
            self._save(stats, self.out_dir / f'ratio_dist_ep{epoch:04d}', M=M, T=T, epoch=epoch)
            if self.accumulate and self._acc is not None:
                avg_stats = self._derive(self._acc)
                self._save(avg_stats, self.out_dir / 'ratio_dist_avg', M=M, T=T, epoch=epoch,
                           title_suffix=' (cumulative avg)')

        # Summary scalars for WandB (use this measurement, not the cumulative).
        rm = stats['ratio_mean']
        finite = np.isfinite(rm)
        return {
            'ratio_diag/ratio_mean_overall': float(np.nanmean(rm[finite])) if finite.any() else math.nan,
            'ratio_diag/ratio_mean_min': float(np.nanmin(rm[finite])) if finite.any() else math.nan,
            'ratio_diag/ratio_var_max': float(np.nanmax(stats['ratio_var'][finite])) if finite.any() else math.nan,
            'ratio_diag/frac_steps_mean_below_0p95': float(np.mean(rm[finite] < 0.95)) if finite.any() else math.nan,
            'ratio_diag/clipfrac_overall': float(np.nanmean(stats['clipfrac'][finite])) if finite.any() else math.nan,
        }

    @staticmethod
    def _derive(acc: dict) -> dict:
        cnt = np.maximum(acc['count'], 1.0)
        logratio_mean = acc['sum_d'] / cnt
        logratio_var = np.maximum(acc['sumsq_d'] / cnt - logratio_mean ** 2, 0.0)
        ratio_mean = acc['sum_r'] / cnt
        ratio_var = np.maximum(acc['sumsq_r'] / cnt - ratio_mean ** 2, 0.0)
        clipfrac = acc['clip_count'] / cnt
        # Steps with no samples -> NaN so they are dropped from plots/summaries.
        empty = acc['count'] < 0.5
        for arr in (logratio_mean, ratio_mean, ratio_var, clipfrac):
            arr[empty] = math.nan
        return {
            't': acc['t'],
            'ratio_mean': ratio_mean,
            'ratio_var': ratio_var,
            'logratio_mean': logratio_mean,
            'logratio_std': np.sqrt(logratio_var),
            'clipfrac': clipfrac,
            'count': acc['count'],
        }

    def _save(self, stats: dict, stem: Path, M: int, T: int, epoch: int,
              title_suffix: str = ''):
        np.savez(str(stem) + '.npz', **stats)
        try:
            self._plot(stats, str(stem) + '.png',
                       title=f'Importance-ratio distribution vs t  '
                             f'(epoch {epoch}, M={M}/step, clip={self.clip_range}){title_suffix}')
        except Exception as e:
            print(f"[ratio-diag] plot failed for {stem}: {e}")

    @staticmethod
    def _plot(stats: dict, out_path: str, title: str = ''):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t = stats['t']
        order = np.argsort(t)[::-1]            # noisy (large t) -> clean (small t)
        x = t[order]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), squeeze=False)

        ax = axes[0][0]
        ax.plot(x, stats['ratio_mean'][order], marker='.', ms=3)
        ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.6)
        ax.set_title('ratio mean  E[r]  (target = 1; <1 ⇒ left-shift)', fontsize=10)

        ax = axes[0][1]
        ax.plot(x, np.sqrt(stats['ratio_var'][order]), marker='.', ms=3, color='tab:orange')
        ax.set_title('ratio std  √Var(r)  (per-step noisiness)', fontsize=10)

        ax = axes[0][2]
        ax.plot(x, stats['clipfrac'][order], marker='.', ms=3, color='tab:red')
        ax.set_title('per-step clipfrac  P(|r−1| > ε)', fontsize=10)

        for a in axes[0]:
            a.set_xlabel('diffusion timestep t  (large = noisy → small = clean)')
            a.invert_xaxis()
            a.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path


def attach_ratio_logging(algo, every_n_epochs: int = 5, out_dir=None,
                         clip_range: float | None = None,
                         logratio_clamp: float = 30.0,
                         accumulate: bool = True) -> RatioDistributionLogger:
    """Attach the per-step ratio-distribution logger to a live PPOAlgorithm.

    Returns the handle; call handle.detach() to remove it. See module docstring.
    """
    return RatioDistributionLogger(
        algo, every_n_epochs=every_n_epochs, out_dir=out_dir,
        clip_range=clip_range, logratio_clamp=logratio_clamp,
        accumulate=accumulate,
    ).attach()
