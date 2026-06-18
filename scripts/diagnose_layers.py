#!/usr/bin/env python
"""
Standalone layer-wise diagnostics for TargetDiff or DiffSBDD PPO policies.

Answers two questions when PPO won't shift the mean reward for *any* reward:

  1. WHERE can I control the network?  -> per-layer activation update size and
     gradient flow across transformer/EGNN layers (9 for TargetDiff, 5 for DiffSBDD).
  2. WHY are atom-identity rewards (QED/logP/SA/aromatic) dead?  -> decompose the
     policy log-prob into coordinate and atom-type channels and compare both their
     magnitudes and the gradient each sends into the trainable params.

     TargetDiff: log_p_pos is continuous Gaussian, log_p_v is categorical (bounded)
                 -> coordinate channel dominates by ~360,000x in gradient.
     DiffSBDD:   BOTH channels are continuous Gaussian (3 vs atom_nf=10 dims)
                 -> at low noise, |log_p_v| >= |log_p_pos| -> near-balanced or reversed.

It loads a checkpoint, pulls one real pocket batch, samples a small rollout with
the chosen reward, then runs the *actual* PPO loss backward with hooks attached.
Compare a baseline vs a "stuck" fine-tuned checkpoint with --finetuned-ckpt.

Example (TargetDiff)
--------------------
    python scripts/diagnose_layers.py \
        --config configs/targetdiff/crossdocked/qed.yaml \
        --baseline-ckpt /path/to/targetdiff_pretrained.pt \
        --finetuned-ckpt /path/to/stuck_qed.pt \
        --model-type targetdiff --reward custom_qed --out-dir diag_out

Example (DiffSBDD)
------------------
    python scripts/diagnose_layers.py \
        --config configs/crossdocked/base.yaml \
        --baseline-ckpt /path/to/diffsbdd_pretrained.ckpt \
        --model-type diffsbdd --reward custom_qed --out-dir diag_diffsbdd
"""

import os
import sys
import json
import argparse
import math
import types
from pathlib import Path

# --- make project + vendored diffsbdd importable (mirrors scripts/train.py) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(1, str(Path(project_root) / "src" / "models" / "diffsbdd"))

import numpy as np
import torch
import yaml

from src.prism.utils import dict_to_namespace
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule
from src.prism.models.policy_factory import build_targetdiff_policy, build_diffsbdd_policy
from src.prism.reward.factory import get_reward_manager
from src.prism.ppo_tuner.rollout_collector import RolloutCollector
from src.prism.ppo_tuner.rollout_buffer import RolloutBuffer
from src.prism.ppo_tuner.loss import compute_ppo_loss
from tests.ppo_debug_utils import reset_seen_mb_ids
from src.prism.analysis.layer_diagnostics import (
    LayerDiagnostics, EGNNLayerDiagnostics, plot_comparison, plot_logprob_channels,
)


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------

def _load_histogram(args, config):
    """Load the DiffSBDD node-count histogram (size_distribution.npy)."""
    if getattr(args, 'histogram_file', None):
        path = Path(args.histogram_file)
    else:
        path = Path(config.datadir) / 'size_distribution.npy'
    if not path.exists():
        raise FileNotFoundError(
            f"DiffSBDD requires a node histogram at {path}. "
            f"Pass --histogram-file to override the default location."
        )
    return np.load(str(path)).tolist()


def _override_single_reward(config, reward_name):
    """Force a single-objective weighted-sum reward (others zeroed) for a clean
    diagnostic signal. No-op if reward_name is None."""
    if reward_name is None:
        return
    rewards = config.reward_params.rewards
    found = False
    for name in list(vars(rewards).keys()):
        setattr(rewards, name, 1.0 if name == reward_name else 0.0)
        found = found or (name == reward_name)
    if not found:
        setattr(rewards, reward_name, 1.0)
    config.reward_params.aggregation = 'weighted_sum'
    print(f"[diag] Reward overridden to single objective: {reward_name}=1.0")


def _trainable_name_set(policy, freeze_except):
    """Param names that PPO would train under this config (substring match)."""
    return {
        name for name, _ in policy.named_parameters()
        if any(tok in name for tok in freeze_except)
    }


def _apply_requires_grad(policy, trainable_names, all_unfrozen):
    n_train = 0
    for name, p in policy.named_parameters():
        flag = True if all_unfrozen else (name in trainable_names)
        p.requires_grad_(flag)
        n_train += int(flag)
    print(f"[diag] requires_grad set on {n_train} params "
          f"({'ALL (full sensitivity probe)' if all_unfrozen else 'configured trainable set'}).")


# ----------------------------------------------------------------------
# Channel decomposition (mirrors loss._get_log_probs mask re-indexing)
# ----------------------------------------------------------------------

def _channels_for_timestep(policy, minibatch, timestep_idx, total_timesteps):
    """Return (log_p_pos, log_p_v) per molecule for one timestep, grad-tracked.

    Works for both TargetDiff and DiffSBDD as long as the policy implements
    log_p_zs_given_zt_channels().  total_timesteps must be passed explicitly
    (config.model.total_timesteps for TargetDiff; policy.total_timesteps for DiffSBDD).
    """
    z_t = minibatch["latents"][:, timestep_idx]
    z_s = minibatch["next_latents"][:, timestep_idx].detach()
    xh_lig, xh_pock = minibatch["molecules"]
    lig_mask, poc_mask = minibatch["masks"]
    t_int = minibatch["timesteps"][:, timestep_idx].float()
    device = z_t.device

    # Re-index per-atom masks to 0..n_mol-1 (identical to loss._get_log_probs).
    unique_ids, new_lig_mask = torch.unique(lig_mask, return_inverse=True)
    mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
    mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
    new_poc_mask = mapping[poc_mask]

    s_int = torch.clamp(t_int - 1, min=0)
    t = (t_int / total_timesteps).unsqueeze(1)
    s = (s_int / total_timesteps).unsqueeze(1)

    return policy.log_p_zs_given_zt_channels(
        s, t, z_t, z_s, xh_pock, new_lig_mask, new_poc_mask
    )


def _grad_norm(loss, params):
    """L2 norm of autograd grad of `loss` w.r.t. `params` (None-safe)."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True, create_graph=False
    )
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.pow(2).sum().item())
    return math.sqrt(sq)


# ----------------------------------------------------------------------
# Per-checkpoint analysis
# ----------------------------------------------------------------------

def analyse_checkpoint(label, ckpt, config, pocket_batch, args, device, reconstruction_fn):
    print(f"\n{'='*70}\n[diag] Analysing '{label}'  ckpt={ckpt}\n{'='*70}")

    model_type = getattr(args, 'model_type', 'targetdiff')

    if model_type == 'diffsbdd':
        node_histogram = _load_histogram(args, config)
        policy, ddpm_outer, dataset_info = build_diffsbdd_policy(
            config=config, device=device,
            node_histogram=node_histogram, warm_start_checkpoint=ckpt,
        )
        total_timesteps = policy.total_timesteps
        diag_cls = EGNNLayerDiagnostics
        diag_target = policy           # EGNNLayerDiagnostics takes the full policy
        reward_ddpm = ddpm_outer       # outer LigandPocketDDPM for reward manager
        recon_fn = None                # DiffSBDD doesn't need a separate decoder fn
    else:
        policy, dataset_info = build_targetdiff_policy(
            config=config, device=device, warm_start_checkpoint=ckpt,
        )
        total_timesteps = getattr(getattr(config, 'model', None), 'total_timesteps', 1000)
        diag_cls = LayerDiagnostics
        diag_target = policy._model    # LayerDiagnostics takes ScorePosNet3D
        reward_ddpm = policy
        recon_fn = reconstruction_fn

    dataset_info['datadir'] = config.datadir

    # Ensure config.model.total_timesteps is set for compute_ppo_loss
    if not hasattr(config, 'model') or config.model is None:
        config.model = types.SimpleNamespace()
    if not hasattr(config.model, 'total_timesteps'):
        config.model.total_timesteps = total_timesteps

    trainable_names = _trainable_name_set(policy, config.freeze_except)
    _apply_requires_grad(policy, trainable_names, args.all_unfrozen)
    trainable_params = [p for n, p in policy.named_parameters() if n in trainable_names]

    # --- collect a small rollout with the configured reward ---
    reward_manager = get_reward_manager(
        config=config, dataset_info=dataset_info,
        ddpm_module=reward_ddpm, reconstruction_fn=recon_fn,
    )
    collector = RolloutCollector(policy, reward_manager, config)
    rollout = collector.collect(pocket_batch, current_epoch=0,
                                get_ligand_and_pocket_fn=policy.get_ligand_and_pocket)
    if rollout['rewards'].numel() == 0:
        raise RuntimeError(f"[{label}] rollout produced no valid samples.")

    buffer = RolloutBuffer(config)
    buffer.load_rollout_data(rollout)
    buffer.compute_advantages()

    # Guard: if the reward has ~no variance, advantages are ~0 and every
    # gradient column will read ~0 — reflecting degenerate reward, NOT the
    # network. Most common cause: a checkpoint that failed to load (see the
    # missing/unexpected keys printed above) producing all-invalid molecules.
    rstd = rollout['rewards'].std().item()
    rmean = rollout['rewards'].mean().item()
    if rstd < 1e-6:
        print(f"\n[diag][WARNING] '{label}': rollout rewards are ~constant "
              f"(mean={rmean:.3f}, std={rstd:.2e}). Advantages ~0 => |dL/dh|/|dL/dx| "
              f"will be ~0 everywhere. This is a degenerate-reward artifact, not the "
              f"network. Check the checkpoint loaded correctly (missing/unexpected "
              f"keys above) and that the reward varies on this pocket.\n")

    # validate_minibatch keeps module-level "seen molecule ID" state across calls;
    # reset it so analysing a second checkpoint (same IDs 0..N) doesn't false-trip.
    reset_seen_mb_ids()
    minibatch = next(iter(buffer.get_minibatches()))

    seq_len = minibatch["latents"].shape[1]
    probe_idxs = sorted(set(
        int(round(v)) for v in np.linspace(0, seq_len - 1, args.probe_timesteps)
    ))
    print(f"[diag] reward_mean={rollout['rewards'].mean().item():.4f} | "
          f"seq_len={seq_len} | probe timestep indices={probe_idxs}")

    # --- layer fwd/bwd diagnostics over the real PPO loss ---
    diag = diag_cls(diag_target).attach(capture_attention=(model_type == 'targetdiff'))
    policy.train()
    for idx in probe_idxs:
        diag.reset()
        if model_type == 'diffsbdd':
            policy._inner.zero_grad(set_to_none=True)
        else:
            policy._model.zero_grad(set_to_none=True)
        loss, _, _, _ = compute_ppo_loss(policy, minibatch, idx, config)
        loss.backward()
        t_val = int(minibatch["timesteps"][0, idx].item())
        diag.collect_step(label=f"t={t_val}")
    summary = diag.summary()
    diag.remove()

    # --- log-prob channel decomposition (over the configured trainable set) ---
    pos_mags, v_mags, gpos, gv = [], [], [], []
    for idx in probe_idxs:
        if model_type == 'diffsbdd':
            policy._inner.zero_grad(set_to_none=True)
        else:
            policy._model.zero_grad(set_to_none=True)
        log_p_pos, log_p_v = _channels_for_timestep(
            policy, minibatch, idx, total_timesteps)
        pos_mags.append(log_p_pos.mean().item())
        v_mags.append(log_p_v.mean().item())
        if trainable_params:
            gpos.append(_grad_norm(log_p_pos.sum(), trainable_params))
            gv.append(_grad_norm(log_p_v.sum(), trainable_params))
    channels = {
        'log_p_pos_mean': float(np.nanmean(pos_mags)),
        'log_p_v_mean': float(np.nanmean(v_mags)),
        'grad_pos_norm': float(np.nanmean(gpos)) if gpos else math.nan,
        'grad_v_norm': float(np.nanmean(gv)) if gv else math.nan,
        'reward_mean': float(rollout['rewards'].mean().item()),
    }
    # Recommended channel_grad_scale.v (with pos=1.0) to make the atom-type
    # channel's gradient comparable to the coordinate channel's.
    gp, gvn = channels['grad_pos_norm'], channels['grad_v_norm']
    channels['recommended_s_v'] = (gp / gvn) if (gvn and gvn > 0) else math.nan
    return summary, channels


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _print_table(label, summary):
    m = summary['metrics']
    L = summary['num_layers']
    print(f"\n--- per-layer summary: {label} ---")
    hdr = f"{'layer':>5} {'||h||':>9} {'Δh_rel':>9} {'Δx':>9} " \
          f"{'|dL/dh|':>10} {'|dL/dx|':>10} {'attnH_x2h':>10} {'attnH_h2x':>10}"
    print(hdr)
    for l in range(L):
        print(f"{l:>5} "
              f"{m['act_h_norm'][l]:>9.3f} "
              f"{m['delta_h_rel'][l]:>9.3f} "
              f"{m['delta_x_norm'][l]:>9.3f} "
              f"{m['grad_h_norm'][l]:>10.2e} "
              f"{m['grad_x_norm'][l]:>10.2e} "
              f"{m['attn_entropy_x2h'][l]:>10.3f} "
              f"{m['attn_entropy_h2x'][l]:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Path to the training YAML.')
    ap.add_argument('--baseline-ckpt', required=True,
                    help='Pretrained checkpoint (.pt/.ckpt).')
    ap.add_argument('--finetuned-ckpt', default=None,
                    help='Optional stuck fine-tuned checkpoint to overlay.')
    ap.add_argument('--model-type', default='targetdiff',
                    choices=['targetdiff', 'diffsbdd'],
                    help='Architecture of the checkpoint (default: targetdiff).')
    ap.add_argument('--histogram-file', default=None,
                    help='Path to size_distribution.npy for DiffSBDD. '
                         'Default: {datadir}/size_distribution.npy.')
    ap.add_argument('--reward', default='custom_qed',
                    help="Single reward to isolate (default custom_qed). "
                         "Pass '' to keep the config's reward weights as-is.")
    ap.add_argument('--num-samples', type=int, default=16,
                    help='Molecules to sample for the rollout (override ppo.n_steps).')
    ap.add_argument('--probe-timesteps', type=int, default=3,
                    help='How many diffusion steps to probe (spread across the chain).')
    ap.add_argument('--all-unfrozen', action='store_true',
                    help='Probe gradient flow as if every layer were trainable '
                         '(full sensitivity map — best for deciding what to unfreeze).')
    ap.add_argument('--datadir', default=None, help='Override config.datadir.')
    ap.add_argument('--out-dir', default='diag_out')
    ap.add_argument('--device', default=None, help="e.g. 'cuda:0' or 'cpu'.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = dict_to_namespace(yaml.safe_load(f))
    if args.datadir:
        config.datadir = args.datadir
    _override_single_reward(config, args.reward or None)

    # Shrink the rollout for fast iteration.
    config.ppo.n_steps = args.num_samples
    config.ppo.batch_size = max(args.num_samples, getattr(config.ppo, 'batch_size', args.num_samples))

    device = torch.device(args.device) if args.device else \
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[diag] device={device}")

    # One real pocket batch.
    dm = LigandPocketDataModule(config)
    dm.setup('fit')
    pocket_batch = next(iter(dm.train_dataloader()))
    pocket_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in pocket_batch.items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # TargetDiff rebuilds RDKit molecules with its own 13-class aromatic decoder;
    # use the same converter the trainer uses so rewards are meaningful.
    # DiffSBDD uses its own built-in decoder — no separate reconstruction_fn needed.
    reconstruction_fn = None
    if args.model_type == 'targetdiff':
        from src.prism.models.targetdiff_inference import make_targetdiff_reconstruction_fn
        reconstruction_fn = make_targetdiff_reconstruction_fn()

    print(f"[diag] model_type={args.model_type}")

    runs = [('baseline', args.baseline_ckpt)]
    if args.finetuned_ckpt:
        runs.append(('finetuned', args.finetuned_ckpt))

    summaries, channels = {}, {}
    for label, ckpt in runs:
        summ, chan = analyse_checkpoint(label, ckpt, config, pocket_batch, args, device, reconstruction_fn)
        summaries[label] = summ
        channels[label] = chan
        _print_table(label, summ)
        print(f"\n[diag] {label} log-prob channels: "
              f"|log_p_pos|={abs(chan['log_p_pos_mean']):.3f}  "
              f"|log_p_v|={abs(chan['log_p_v_mean']):.3f}  "
              f"grad_pos={chan['grad_pos_norm']:.2e}  grad_v={chan['grad_v_norm']:.2e}")
        rec = chan.get('recommended_s_v', math.nan)
        if not math.isnan(rec):
            print(f"[diag] {label} -> coords currently outweigh atom-types "
                  f"{rec:.3g}:1 in gradient.\n"
                  f"        To balance them, set in your config:\n"
                  f"          ppo:\n"
                  f"            channel_grad_scale:\n"
                  f"              pos: 1.0\n"
                  f"              v:   {rec:.3g}")

    title = f"PRISM layer diagnostics — {args.model_type} ({args.reward or 'config reward'})"
    fig1 = plot_comparison(summaries, str(out / 'layer_diagnostics.png'), title=title)
    fig2 = plot_logprob_channels(channels, str(out / 'logprob_channels.png'), title=title)
    with open(out / 'diagnostics.json', 'w') as f:
        json.dump({'summaries': summaries, 'channels': channels,
                   'args': vars(args)}, f, indent=2)

    print(f"\n[diag] DONE. Wrote:\n  {fig1}\n  {fig2}\n  {out/'diagnostics.json'}")


if __name__ == '__main__':
    main()
