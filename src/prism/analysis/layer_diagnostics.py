"""
Layer-wise diagnostics for the TargetDiff UniTransformer backbone.

Purpose
-------
When PPO fine-tuning fails to shift the mean reward *regardless of which reward*
is used, the problem is almost always upstream of the reward: the learning
signal either never reaches the trainable layers, or the trainable layers have
no capacity to change the generated molecule. This module instruments the
network so that failure mode becomes visible.

For each of the 9 transformer layers (``refine_net.base_block[l]``) it records,
per forward/backward pass:

  * Activation scale         ``||h_l||``, ``||x_l||``   (mean over ligand atoms)
  * Activation update        ``||Δh_l|| / ||h_{l-1}||``, ``||Δx_l||``
        -> which layers actually transform the molecule (which to unfreeze)
  * Gradient flow            ``||dL/dh_l||``, ``||dL/dx_l||``
        -> where the backward learning signal reaches vs. vanishes (da/dc, da/dx)
  * Attention entropy        mean per-node entropy of the x2h / h2x attention
        -> whether a layer does focused relational work or just passes through

It does **not** mutate model behaviour: activation/grad stats come from hooks,
and attention capture is an opt-in flag (default off) on the attention sublayers.

Usage
-----
    diag = LayerDiagnostics(score_model)        # score_model = policy._model
    diag.attach(capture_attention=True)
    for t in probe_timesteps:
        diag.reset()
        loss = ...                              # one PPO-loss forward
        loss.backward()
        diag.collect_step(label=f"t={t}")
    summary = diag.summary()
    diag.remove()                               # restore the model exactly

To see the *full* sensitivity profile (gradient flow as if every layer were
trainable), set ``requires_grad=True`` on all params before the backward — the
script does this in its "probe" mode.
"""

from __future__ import annotations

import copy
import math

import torch


def _row_norm_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean L2 row-norm over the masked (ligand) rows. NaN-safe for empty mask."""
    if t is None:
        return math.nan
    sel = t[mask]
    if sel.numel() == 0:
        return math.nan
    return sel.norm(dim=-1).mean().item()


class LayerDiagnostics:
    """Attach to a ScorePosNet3D model and record per-layer fwd/bwd statistics."""

    # Metric keys produced per layer (kept in one place so plotting stays in sync).
    METRIC_KEYS = (
        'act_h_norm', 'act_x_norm',
        'delta_h_rel', 'delta_x_norm',
        'grad_h_norm', 'grad_x_norm',
        'attn_entropy_x2h', 'attn_entropy_h2x',
    )

    def __init__(self, score_model):
        # score_model is the inner ScorePosNet3D (policy._model). Its UniTransformer
        # backbone lives at .refine_net with the per-layer ModuleList .base_block.
        self.model = score_model
        self.layers = list(score_model.refine_net.base_block)
        self.num_layers = len(self.layers)

        self._handles = []
        self._attn_layers = []          # (x2h_sublayer, h2x_sublayer) per layer
        self._capture_attention = False

        self._cur = {}                  # per-layer scratch for the current step
        self.records = []               # list of {'label', 'layers': {l: {...}}}

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, capture_attention: bool = True):
        """Register forward hooks (and toggle attention capture). Idempotent-safe
        as long as remove() is called first."""
        self._capture_attention = capture_attention
        for l_idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_hook(self._make_forward_hook(l_idx))
            )
            # The attention entropy is computed inside the x2h/h2x sublayers; turn
            # capture on so they stash _last_attn_entropy each forward.
            x2h = layer.x2h_layers[0] if len(layer.x2h_layers) else None
            h2x = layer.h2x_layers[0] if len(layer.h2x_layers) else None
            self._attn_layers.append((x2h, h2x))
            if capture_attention:
                if x2h is not None:
                    x2h.capture_attention = True
                if h2x is not None:
                    h2x.capture_attention = True
        return self

    def remove(self):
        """Remove all hooks and restore the model to its original state."""
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._capture_attention:
            for x2h, h2x in self._attn_layers:
                if x2h is not None:
                    x2h.capture_attention = False
                    x2h._last_attn_entropy = None
                if h2x is not None:
                    h2x.capture_attention = False
                    h2x._last_attn_entropy = None
        self._attn_layers = []
        self._cur = {}

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_forward_hook(self, l_idx: int):
        def hook(module, inputs, output):
            # Positional call: layer(h, x, edge_attr, edge_index, mask_ligand, ...)
            in_h, in_x = inputs[0], inputs[1]
            mask_ligand = inputs[4]
            out_h, out_x = output[0], output[1]

            mask = mask_ligand.bool()

            in_h_norm = _row_norm_mean(in_h, mask)
            delta_h = (out_h - in_h)
            # Relative update size, atom-wise then averaged: ||Δh_i|| / (||h_i|| + eps)
            sel = mask
            num = delta_h[sel].norm(dim=-1)
            den = in_h[sel].norm(dim=-1) + 1e-8
            delta_h_rel = (num / den).mean().item() if num.numel() else math.nan

            rec = {
                'act_h_norm': _row_norm_mean(out_h, mask),
                'act_x_norm': _row_norm_mean(out_x, mask),
                'delta_h_rel': delta_h_rel,
                'delta_x_norm': _row_norm_mean(out_x - in_x, mask),
                'attn_entropy_x2h': math.nan,
                'attn_entropy_h2x': math.nan,
                'grad_h_norm': math.nan,
                'grad_x_norm': math.nan,
                '_mask': mask,
            }

            if self._capture_attention:
                # Attention-entropy capture requires an in-forward hook the vendored
                # UniTransformer doesn't expose, so these stay NaN unless a layer
                # opts in via a _last_attn_entropy attribute. getattr keeps this
                # robust against the stock (unmodified) vendor module.
                x2h, h2x = self._attn_layers[l_idx]
                x2h_ent = getattr(x2h, '_last_attn_entropy', None) if x2h is not None else None
                h2x_ent = getattr(h2x, '_last_attn_entropy', None) if h2x is not None else None
                if x2h_ent is not None:
                    rec['attn_entropy_x2h'] = float(x2h_ent)
                if h2x_ent is not None:
                    rec['attn_entropy_h2x'] = float(h2x_ent)

            self._cur[l_idx] = rec

            # Capture dL/d(layer output) for h and x — only if they are part of a
            # grad-tracking graph (true when upstream params require grad).
            if out_h.requires_grad:
                out_h.register_hook(self._make_grad_hook(l_idx, 'grad_h_norm'))
            if out_x.requires_grad:
                out_x.register_hook(self._make_grad_hook(l_idx, 'grad_x_norm'))

        return hook

    def _make_grad_hook(self, l_idx: int, key: str):
        def grad_hook(grad):
            rec = self._cur.get(l_idx)
            if rec is not None:
                rec[key] = _row_norm_mean(grad, rec['_mask'])
            return None  # do not modify the gradient
        return grad_hook

    # ------------------------------------------------------------------
    # Step accumulation
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the per-step scratch. Call before each forward+backward."""
        self._cur = {}

    def collect_step(self, label: str = ''):
        """Snapshot the current step's per-layer metrics into the record log.

        Call after backward() so gradient hooks have fired. Drops the transient
        mask tensor so records stay light and picklable.
        """
        layers = {}
        for l_idx in range(self.num_layers):
            rec = self._cur.get(l_idx, {})
            layers[l_idx] = {k: rec.get(k, math.nan) for k in self.METRIC_KEYS}
        self.records.append({'label': label, 'layers': layers})
        self.reset()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Average each metric across all recorded steps, per layer.

        Returns {'num_layers', 'steps', 'metrics': {key: [per-layer floats]}}.
        """
        metrics = {k: [math.nan] * self.num_layers for k in self.METRIC_KEYS}
        for l_idx in range(self.num_layers):
            for key in self.METRIC_KEYS:
                vals = [
                    r['layers'][l_idx][key]
                    for r in self.records
                    if not math.isnan(r['layers'][l_idx].get(key, math.nan))
                ]
                metrics[key][l_idx] = (sum(vals) / len(vals)) if vals else math.nan
        return {
            'num_layers': self.num_layers,
            'steps': [r['label'] for r in self.records],
            'metrics': metrics,
        }


# ----------------------------------------------------------------------
# DiffSBDD / EGNN diagnostics
# ----------------------------------------------------------------------

class EGNNLayerDiagnostics:
    """Attach to a DiffSBDDPolicy and record per-EquivariantBlock fwd/bwd statistics.

    Mirrors LayerDiagnostics for TargetDiff but targets the EGNN backbone
    (policy._inner.dynamics.egnn.e_block_0 … e_block_{n-1}).

    Key differences from LayerDiagnostics:
    - EGNN concatenates ligand + pocket atoms before the message-passing loop,
      so hooks see ALL atoms and no ligand mask is available.  Norms are
      therefore reported over all atoms (still diagnostic for channel balance).
    - GCL uses sigmoid scalar attention (not softmax), so attn_entropy_* is
      always NaN.
    """

    METRIC_KEYS = LayerDiagnostics.METRIC_KEYS

    def __init__(self, policy):
        egnn = policy._inner.dynamics.egnn
        self.num_layers = egnn.n_layers
        self.layers = [egnn._modules[f"e_block_{i}"] for i in range(self.num_layers)]
        self._handles = []
        self._cur = {}
        self.records = []

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, capture_attention: bool = False):
        """Register forward hooks. capture_attention is accepted but ignored
        (EGNN has no entropy-meaningful attention)."""
        for l_idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_hook(self._make_forward_hook(l_idx))
            )
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._cur = {}

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_forward_hook(self, l_idx: int):
        def hook(module, inputs, output):
            # EquivariantBlock.forward(h, x, edge_index, ...) → (h, x)
            in_h = inputs[0]   # [N_all_atoms, hidden_nf]
            in_x = inputs[1]   # [N_all_atoms, 3]
            out_h, out_x = output[0], output[1]

            num = (out_h - in_h).norm(dim=-1)
            den = in_h.norm(dim=-1) + 1e-8
            rec = {
                'act_h_norm':       out_h.norm(dim=-1).mean().item(),
                'act_x_norm':       out_x.norm(dim=-1).mean().item(),
                'delta_h_rel':      (num / den).mean().item(),
                'delta_x_norm':     (out_x - in_x).norm(dim=-1).mean().item(),
                'attn_entropy_x2h': math.nan,
                'attn_entropy_h2x': math.nan,
                'grad_h_norm':      math.nan,
                'grad_x_norm':      math.nan,
            }
            self._cur[l_idx] = rec

            if out_h.requires_grad:
                out_h.register_hook(self._make_grad_hook(l_idx, 'grad_h_norm'))
            if out_x.requires_grad:
                out_x.register_hook(self._make_grad_hook(l_idx, 'grad_x_norm'))

        return hook

    def _make_grad_hook(self, l_idx: int, key: str):
        def grad_hook(grad):
            rec = self._cur.get(l_idx)
            if rec is not None:
                rec[key] = grad.norm(dim=-1).mean().item()
            return None
        return grad_hook

    # ------------------------------------------------------------------
    # Step accumulation — identical interface to LayerDiagnostics
    # ------------------------------------------------------------------

    def reset(self):
        self._cur = {}

    def collect_step(self, label: str = ''):
        layers = {}
        for l_idx in range(self.num_layers):
            rec = self._cur.get(l_idx, {})
            layers[l_idx] = {k: rec.get(k, math.nan) for k in self.METRIC_KEYS}
        self.records.append({'label': label, 'layers': layers})
        self.reset()

    def summary(self) -> dict:
        metrics = {k: [math.nan] * self.num_layers for k in self.METRIC_KEYS}
        for l_idx in range(self.num_layers):
            for key in self.METRIC_KEYS:
                vals = [
                    r['layers'][l_idx][key]
                    for r in self.records
                    if not math.isnan(r['layers'][l_idx].get(key, math.nan))
                ]
                metrics[key][l_idx] = (sum(vals) / len(vals)) if vals else math.nan
        return {
            'num_layers': self.num_layers,
            'steps': [r['label'] for r in self.records],
            'metrics': metrics,
        }


# ----------------------------------------------------------------------
# Per-layer decision attribution (logit-lens) — TargetDiff only
# ----------------------------------------------------------------------

class TargetDiffDecisionLens:
    """Logit-lens across the 9 UniTransformer layers (TargetDiff).

    Standard layer diagnostics measure activation/grad *magnitude*. This probe
    measures the *decision variables* instead: it grabs each base_block's ligand
    hidden ``h_l`` and coords ``x_l``, applies the model's frozen atom-type head
    (``v_inference``) to ``h_l``, and asks where along depth the molecule is
    actually decided:

      * ``atomtype_entropy[l]``  – entropy of softmax(v_inference(h_l)); how
        uncertain the element identity still is at layer l.
      * ``atomtype_kl_final[l]`` – KL(p_l ‖ p_final); 0 ⇒ the atom-type decision
        is already made by layer l (later layers add nothing).
      * ``coord_drift[l]``       – ‖x_l − x_final‖; 0 ⇒ coordinates have settled
        by layer l.

    If both curves flatten in the *frozen* early layers, unfreezing the last K
    layers cannot move the output — which is exactly the layer-agnostic flatness
    we are chasing. No model mutation: pure forward hooks.
    """

    DECISION_KEYS = ('atomtype_entropy', 'atomtype_kl_final', 'coord_drift')

    def __init__(self, score_model):
        self.model = score_model
        self.head = score_model.v_inference
        self.layers = list(score_model.refine_net.base_block)
        self.num_layers = len(self.layers)
        self._handles = []
        self._cur = {}
        self.records = []

    def attach(self):
        for l_idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_hook(self._make_hook(l_idx)))
        return self

    def _make_hook(self, l_idx: int):
        def hook(module, inputs, output):
            mask = inputs[4].bool()                       # ligand-atom mask
            out_h, out_x = output[0], output[1]
            logp = torch.log_softmax(self.head(out_h[mask]), dim=-1).detach()
            self._cur[l_idx] = {'logp': logp, 'x': out_x[mask].detach()}
        return hook

    def reset(self):
        self._cur = {}

    def collect_step(self, label: str = ''):
        final = self._cur.get(self.num_layers - 1)
        layers = {}
        for l in range(self.num_layers):
            c = self._cur.get(l)
            if c is None:
                layers[l] = {k: math.nan for k in self.DECISION_KEYS}
                continue
            p = c['logp'].exp()
            ent = -(p * c['logp']).sum(dim=-1).mean().item()
            if final is not None:
                kl = (p * (c['logp'] - final['logp'])).sum(dim=-1).mean().item()
                drift = (c['x'] - final['x']).norm(dim=-1).mean().item()
            else:
                kl = drift = math.nan
            layers[l] = {'atomtype_entropy': ent,
                         'atomtype_kl_final': kl, 'coord_drift': drift}
        self.records.append({'label': label, 'layers': layers})
        self.reset()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._cur = {}

    def summary(self) -> dict:
        metrics = {k: [math.nan] * self.num_layers for k in self.DECISION_KEYS}
        for l in range(self.num_layers):
            for k in self.DECISION_KEYS:
                vals = [r['layers'][l][k] for r in self.records
                        if not math.isnan(r['layers'][l].get(k, math.nan))]
                metrics[k][l] = (sum(vals) / len(vals)) if vals else math.nan
        return {'num_layers': self.num_layers,
                'steps': [r['label'] for r in self.records],
                'metrics': metrics}


def plot_layer_decision(summaries: dict, out_path: str, title: str = ''):
    """Plot the logit-lens decision curves across layers (one line per label)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('atomtype_entropy', 'Atom-type entropy  H(softmax(v_inference(h_l)))'),
        ('atomtype_kl_final', 'Atom-type KL(p_l ‖ p_final)  (0 ⇒ decided by layer l)'),
        ('coord_drift', 'Coord drift  ||x_l − x_final||  (0 ⇒ coords settled)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), squeeze=False)
    for idx, (key, lbl) in enumerate(panels):
        ax = axes[0][idx]
        for run_label, summ in summaries.items():
            ys = summ['metrics'].get(key, [])
            ax.plot(range(len(ys)), ys, marker='o', markersize=4, label=run_label)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('layer index (0 = first … last)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_trajectory(traj: dict, out_path: str, title: str = ''):
    """Plot per-timestep denoising-trajectory curves (one line per label).

    Each label maps to a dict of equal-length lists keyed by:
        't', 'log_p_pos', 'log_p_v', 'grad_pos', 'grad_v',
        'pos_sigma', 'atomtype_entropy'
    x-axis is the diffusion timestep t (large = noisy, small = near-clean).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [
        (('log_p_pos', 'log_p_v'), '|log-prob| per channel', True, 'abs'),
        (('grad_pos', 'grad_v'), '||∂channel/∂θ|| into trainable params', True, None),
        (('grad_ratio',), 'grad_pos / grad_v   (>1 ⇒ coords dominate)', True, None),
        (('pos_sigma',), 'coord σ(t)  (malleability; small ⇒ frozen)', True, None),
        (('atomtype_entropy',), 'atom-type entropy  (small ⇒ identity locked)', False, None),
        (('log_p_v',), '|log_p_v| alone (atom-type signal)', True, 'abs'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), squeeze=False)
    for idx, (keys, lbl, logy, mode) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        for run_label, d in traj.items():
            ts = d.get('t', [])
            order = np.argsort(ts)[::-1]      # plot noisy→clean (large t first)
            xs = np.asarray(ts)[order]
            for k in keys:
                ys = np.asarray(d.get(k, [math.nan] * len(ts)), dtype=float)[order]
                if mode == 'abs':
                    ys = np.abs(ys)
                lab = f"{run_label}:{k}" if len(traj) * len(keys) > 1 else k
                ax.plot(xs, ys, marker='.', markersize=4, label=lab)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('diffusion timestep t  (large = noisy → small = near-clean)')
        ax.invert_xaxis()                      # clean on the right
        ax.grid(True, alpha=0.3)
        if logy:
            try:
                ax.set_yscale('log')
            except Exception:
                pass
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Plotting — comparison across one or more labelled summaries
# ----------------------------------------------------------------------

_PLOT_PANELS = [
    ('act_h_norm', 'Activation norm  ||h_l||  (ligand)'),
    ('act_x_norm', 'Activation norm  ||x_l||  (ligand)'),
    ('delta_h_rel', 'Relative update  ||Δh_l|| / ||h_{l-1}||'),
    ('delta_x_norm', 'Coord update  ||Δx_l||'),
    ('grad_h_norm', 'Grad flow  ||dL/dh_l||  (da/dc)'),
    ('grad_x_norm', 'Grad flow  ||dL/dx_l||  (da/dx)'),
    ('attn_entropy_x2h', 'Attention entropy (x2h)'),
    ('attn_entropy_h2x', 'Attention entropy (h2x)'),
]


def plot_comparison(summaries: dict, out_path: str, title: str = ''):
    """Plot per-layer diagnostics for one or more labelled summaries.

    Args:
        summaries: {label -> summary dict from LayerDiagnostics.summary()}.
        out_path:  PNG path to save.
        title:     Optional figure suptitle.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(_PLOT_PANELS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), squeeze=False)

    for idx, (key, label) in enumerate(_PLOT_PANELS):
        ax = axes[idx // ncols][idx % ncols]
        for run_label, summ in summaries.items():
            ys = summ['metrics'].get(key, [])
            xs = list(range(len(ys)))
            ax.plot(xs, ys, marker='o', markersize=4, label=run_label)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('layer index (0 = first, 8 = last)')
        ax.grid(True, alpha=0.3)
        if 'grad' in key or 'norm' in key:
            # gradients/norms can span orders of magnitude
            try:
                ax.set_yscale('log')
            except Exception:
                pass
        if idx == 0:
            ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_logprob_channels(channels: dict, out_path: str, title: str = ''):
    """Bar chart of the log-prob channel decomposition.

    Args:
        channels: {label -> {'log_p_pos_mean', 'log_p_v_mean',
                             'grad_pos_norm', 'grad_v_norm'}}.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(channels.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    x = np.arange(len(labels))
    w = 0.38

    pos_mag = [abs(channels[l].get('log_p_pos_mean', math.nan)) for l in labels]
    v_mag = [abs(channels[l].get('log_p_v_mean', math.nan)) for l in labels]
    ax1.bar(x - w / 2, pos_mag, w, label='|log_p_pos| (coords)')
    ax1.bar(x + w / 2, v_mag, w, label='|log_p_v| (atom types)')
    ax1.set_yscale('log')
    ax1.set_title('Log-prob magnitude per channel\n(if pos >> v, atom-type rewards are starved)')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    gpos = [channels[l].get('grad_pos_norm', math.nan) for l in labels]
    gv = [channels[l].get('grad_v_norm', math.nan) for l in labels]
    ax2.bar(x - w / 2, gpos, w, label='||∂L/∂θ|| from pos channel')
    ax2.bar(x + w / 2, gv, w, label='||∂L/∂θ|| from v channel')
    ax2.set_yscale('log')
    ax2.set_title('Gradient into trainable params per channel')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
