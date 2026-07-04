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
    TargetDiffDecisionLens, plot_layer_decision, plot_trajectory,
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
# Diagnostic 2 — denoising trajectory (dense per-timestep, NOT averaged)
# ----------------------------------------------------------------------

def run_trajectory(policy, minibatch, n_steps, total_timesteps, trainable_params,
                   model_type, out, label, title):
    """Dense per-timestep sweep: when is the molecule decided, and where is the
    channel gradient healthy. Keeps every timestep (no averaging)."""
    seq_len = minibatch["latents"].shape[1]
    idxs = sorted(set(int(round(v)) for v in np.linspace(0, seq_len - 1, n_steps)))
    has_stats = hasattr(policy, 'trajectory_stats_given_zt')

    rec = {k: [] for k in ('t', 'log_p_pos', 'log_p_v', 'grad_pos', 'grad_v',
                            'grad_ratio', 'pos_sigma', 'atomtype_entropy')}
    for idx in idxs:
        t_val = int(minibatch["timesteps"][0, idx].item())
        if has_stats:
            # one forward → channels + malleability signals
            z_t = minibatch["latents"][:, idx]
            z_s = minibatch["next_latents"][:, idx].detach()
            _, xh_pock = minibatch["molecules"]
            lig_mask, poc_mask = minibatch["masks"]
            t_int = minibatch["timesteps"][:, idx].float()
            device = z_t.device
            unique_ids, new_lig = torch.unique(lig_mask, return_inverse=True)
            mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
            mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
            new_poc = mapping[poc_mask]
            t = (t_int / total_timesteps).unsqueeze(1)
            s = (torch.clamp(t_int - 1, min=0) / total_timesteps).unsqueeze(1)
            d = policy.trajectory_stats_given_zt(s, t, z_t, z_s, xh_pock, new_lig, new_poc)
            lp_pos, lp_v = d['log_p_pos'], d['log_p_v']
            sigma = d['pos_sigma'].mean().item()
            ent = d['atomtype_entropy'].mean().item()
        else:
            lp_pos, lp_v = _channels_for_timestep(policy, minibatch, idx, total_timesteps)
            sigma = ent = math.nan
        gp = _grad_norm(lp_pos.sum(), trainable_params) if trainable_params else math.nan
        gv = _grad_norm(lp_v.sum(), trainable_params) if trainable_params else math.nan
        rec['t'].append(t_val)
        rec['log_p_pos'].append(lp_pos.mean().item())
        rec['log_p_v'].append(lp_v.mean().item())
        rec['grad_pos'].append(gp)
        rec['grad_v'].append(gv)
        rec['grad_ratio'].append((gp / gv) if (gv and gv > 0) else math.nan)
        rec['pos_sigma'].append(sigma)
        rec['atomtype_entropy'].append(ent)

    png = plot_trajectory({label: rec}, str(out / f'trajectory_{label}.png'), title=title)
    print(f"\n--- denoising trajectory: {label}  ({len(idxs)} steps) ---")
    print(f"  probed t∈[{min(rec['t'])}, {max(rec['t'])}]; wrote {png}")
    # Flag the t-region where the atom-type channel gradient is healthiest.
    gr = np.array(rec['grad_ratio'], dtype=float)
    if np.isfinite(gr).any():
        best = int(np.nanargmin(gr))
        print(f"  grad_pos/grad_v is smallest (atom-type channel relatively strongest) "
              f"at t≈{rec['t'][best]} (ratio={gr[best]:.3g}).")
    return rec


# ----------------------------------------------------------------------
# Diagnostic 3 — per-layer decision attribution (logit-lens, TargetDiff)
# ----------------------------------------------------------------------

def run_layer_decision(policy, minibatch, probe_idxs, total_timesteps, out, label, title):
    """Where across the 9 layers is the atom-type / coordinate decision made."""
    lens = TargetDiffDecisionLens(policy._model).attach()
    policy.eval()
    with torch.no_grad():
        for idx in probe_idxs:
            lens.reset()
            z_t = minibatch["latents"][:, idx]
            z_s = minibatch["next_latents"][:, idx].detach()
            _, xh_pock = minibatch["molecules"]
            lig_mask, poc_mask = minibatch["masks"]
            t_int = minibatch["timesteps"][:, idx].float()
            device = z_t.device
            unique_ids, new_lig = torch.unique(lig_mask, return_inverse=True)
            mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
            mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
            new_poc = mapping[poc_mask]
            t = (t_int / total_timesteps).unsqueeze(1)
            s = (torch.clamp(t_int - 1, min=0) / total_timesteps).unsqueeze(1)
            policy.log_p_zs_given_zt(s, t, z_t, z_s, xh_pock, new_lig, new_poc)
            lens.collect_step(label=f"t={int(minibatch['timesteps'][0, idx].item())}")
    summ = lens.summary()
    lens.remove()
    policy.train()                      # restore (eval() was only for the lens pass)
    png = plot_layer_decision({label: summ}, str(out / f'layer_decision_{label}.png'), title=title)
    m = summ['metrics']
    print(f"\n--- per-layer decision attribution: {label} ---")
    print(f"{'layer':>5} {'atom_entropy':>13} {'atom_KL_final':>14} {'coord_drift':>12}")
    for l in range(summ['num_layers']):
        print(f"{l:>5} {m['atomtype_entropy'][l]:>13.4f} "
              f"{m['atomtype_kl_final'][l]:>14.4f} {m['coord_drift'][l]:>12.4f}")
    print(f"  wrote {png}.  If KL→0 and coord_drift→0 in the early (frozen) layers, "
          f"unfreezing\n  the last layers cannot move the output — the molecule is "
          f"decided upstream.")
    return summ


# ----------------------------------------------------------------------
# Diagnostic 4 — geometry coord↔atom coupling
# ----------------------------------------------------------------------

def run_coupling(policy, rollout, minibatch, probe_idxs, config, dataset_info,
                 model_type, label):
    """Is good geometry coupled to atom identity? (a) correlate per-mol reward
    with composition; (b) split the PPO-loss gradient into atom-type head vs the
    rest, to see whether the (coord-based) reward recruits atom-type params."""
    out = {}
    # (a) correlational — reward vs atom-type composition
    xh_lig, _ = rollout['molecules']
    lig_mask = rollout['masks'][0]
    rewards = rollout['rewards']
    n_dims = policy.n_dims
    atom_idx = xh_lig[:, n_dims:].argmax(dim=-1)
    n_types = xh_lig[:, n_dims:].shape[1]
    uniq, new_mask = torch.unique(lig_mask, return_inverse=True)
    n_mol = len(uniq)
    from torch_scatter import scatter_mean as _sm
    decoder = dataset_info.get('atom_decoder') if isinstance(dataset_info, dict) else None
    corrs = []
    for k in range(n_types):
        frac = _sm((atom_idx == k).float(), new_mask, dim=0, dim_size=n_mol)
        if frac.std() < 1e-8 or rewards.std() < 1e-8:
            continue
        c = float(torch.corrcoef(torch.stack([frac, rewards.float()]))[0, 1].item())
        name = decoder[k] if (decoder and k < len(decoder)) else f"type_{k}"
        corrs.append((name, c, float(frac.mean().item())))
    corrs.sort(key=lambda r: -abs(r[1]))
    out['reward_composition_corr'] = corrs

    # (b) gradient split — atom-type head (v_inference) vs the rest
    head_sq = rest_sq = 0.0
    if model_type == 'targetdiff':
        idx = probe_idxs[-1]                      # a low-noise step
        policy._model.zero_grad(set_to_none=True)
        loss, _, _, _ = compute_ppo_loss(policy, minibatch, idx, config)
        loss.backward()
        for n, p in policy.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.pow(2).sum().item())
            if 'v_inference' in n:
                head_sq += g
            else:
                rest_sq += g
        policy._model.zero_grad(set_to_none=True)
    head_norm, rest_norm = math.sqrt(head_sq), math.sqrt(rest_sq)
    total = head_norm + rest_norm
    out['grad_atomtype_head_norm'] = head_norm
    out['grad_rest_norm'] = rest_norm
    out['grad_atomtype_head_frac'] = (head_norm / total) if total > 0 else math.nan

    print(f"\n--- geometry coord↔atom coupling: {label} ---")
    if corrs:
        print("  reward vs atom-type-fraction correlation (|r| desc):")
        for name, c, mfrac in corrs[:6]:
            print(f"    {name:>8}: r={c:+.3f}  (mean frac={mfrac:.3f})")
        print("  large |r| ⇒ the (coord-based) reward depends on which atoms are present "
              "⇒ coords-only\n  control is insufficient.")
    else:
        print("  (composition or reward had ~no variance; correlation skipped)")
    if model_type == 'targetdiff':
        print(f"  PPO-grad share: atom-type head={out['grad_atomtype_head_frac']:.3f} "
              f"(|g|={head_norm:.2e}) vs rest (|g|={rest_norm:.2e}).")
    return out


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

    # --- optional extra diagnostics (gated by flags) ---
    out = Path(args.out_dir)
    title = f"{args.model_type} ({args.reward or 'config reward'}) — {label}"
    extra = {}
    if getattr(args, 'trajectory', False):
        extra['trajectory'] = run_trajectory(
            policy, minibatch, args.trajectory_steps, total_timesteps,
            trainable_params, model_type, out, label, title)
    if getattr(args, 'layer_decision', False):
        if model_type == 'targetdiff':
            extra['layer_decision'] = run_layer_decision(
                policy, minibatch, probe_idxs, total_timesteps, out, label, title)
        else:
            print(f"[diag] --layer-decision is TargetDiff-only "
                  f"(EGNN has no separate atom-type head); skipping for {label}.")
    if getattr(args, 'coupling', False):
        extra['coupling'] = run_coupling(
            policy, rollout, minibatch, probe_idxs, config, dataset_info,
            model_type, label)

    return summary, channels, extra


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
                    help='How many diffusion steps to probe (spread across the chain) '
                         'for the layer/channel summary and decision lens.')
    # --- optional diagnostic modes (default off; plain run is unchanged) ---
    ap.add_argument('--trajectory', action='store_true',
                    help='Dense per-timestep denoising sweep (when is the molecule decided).')
    ap.add_argument('--trajectory-steps', type=int, default=100,
                    help='Timesteps for the --trajectory sweep (default 100).')
    ap.add_argument('--layer-decision', action='store_true',
                    help='Logit-lens: where across the 9 layers the molecule is decided (TargetDiff).')
    ap.add_argument('--coupling', action='store_true',
                    help='Geometry coord↔atom coupling: reward/composition corr + grad split.')
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

    summaries, channels, extras = {}, {}, {}
    for label, ckpt in runs:
        summ, chan, extra = analyse_checkpoint(
            label, ckpt, config, pocket_batch, args, device, reconstruction_fn)
        summaries[label] = summ
        channels[label] = chan
        extras[label] = extra
        _print_table(label, summ)
        print(f"\n[diag] {label} log-prob channels: "
              f"|log_p_pos|={abs(chan['log_p_pos_mean']):.3f}  "
              f"|log_p_v|={abs(chan['log_p_v_mean']):.3f}  "
              f"grad_pos={chan['grad_pos_norm']:.2e}  grad_v={chan['grad_v_norm']:.2e}")
        gp, gvn = chan['grad_pos_norm'], chan['grad_v_norm']
        if gvn and gvn > 0 and not math.isnan(gp):
            print(f"[diag] {label} -> coordinate channel outweighs atom-types "
                  f"{gp / gvn:.3g}:1 in gradient (measurement only).")

    title = f"PRISM layer diagnostics — {args.model_type} ({args.reward or 'config reward'})"
    fig1 = plot_comparison(summaries, str(out / 'layer_diagnostics.png'), title=title)
    fig2 = plot_logprob_channels(channels, str(out / 'logprob_channels.png'), title=title)
    written = [fig1, fig2]

    # Overlay the extra diagnostics across all labels (baseline vs finetuned).
    if args.trajectory and any('trajectory' in e for e in extras.values()):
        traj = {l: e['trajectory'] for l, e in extras.items() if 'trajectory' in e}
        written.append(plot_trajectory(traj, str(out / 'trajectory.png'), title=title))
    if args.layer_decision and any('layer_decision' in e for e in extras.values()):
        ld = {l: e['layer_decision'] for l, e in extras.items() if 'layer_decision' in e}
        written.append(plot_layer_decision(ld, str(out / 'layer_decision.png'), title=title))

    with open(out / 'diagnostics.json', 'w') as f:
        json.dump({'summaries': summaries, 'channels': channels,
                   'extras': extras, 'args': vars(args)}, f, indent=2)

    print(f"\n[diag] DONE. Wrote:")
    for p in written:
        print(f"  {p}")
    print(f"  {out/'diagnostics.json'}")


if __name__ == '__main__':
    main()
