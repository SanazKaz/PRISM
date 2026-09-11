"""
Opt-in activation (gradient) checkpointing for TargetDiff's UniTransformer,
applied from PRISM without editing the vendored model.

src/models/targetdiff is vendored upstream code and is never edited directly
(see CLAUDE.md). This used to live as a small in-place change to
uni_transformer.py's UniTransformerO2TwoUpdateGeneral.forward — recomputing
each base_block layer's activations in backward instead of storing them,
cutting the activation-memory peak roughly 30-50% at the cost of one extra
forward per layer per backward pass. That's the mechanism behind the
multi-objective PPO OOM noted for L40s (activation-bound in the update
backprop, not docking/rewards).

Reproduced here as an external patch: monkey-patch each base_block layer's
bound forward() method in place, rather than wrapping it in a parent module.
Wrapping would add a name segment to every parameter under it (e.g.
"base_block.0.foo" -> "base_block.0.inner.foo"), silently breaking every
existing .pt/.ckpt trained before the patch existed. Patching the method in
place changes no parameter names and no state_dict keys, so checkpoint
compatibility is untouched regardless of whether this patch is applied in a
given process.

Env-gated (PRISM_GRAD_CKPT=1) so default behaviour is unchanged; read once,
at patch time, matching the original code's self.use_grad_ckpt-at-__init__
behaviour (toggling the env var after a model is built has no effect on that
model — build a fresh one, or restart, to change it).
"""

import os
import types

import torch
from torch.utils.checkpoint import checkpoint


def apply_targetdiff_grad_checkpointing(score_model) -> bool:
    """Patch score_model.refine_net.base_block's layers for activation
    checkpointing, if that attribute exists and PRISM_GRAD_CKPT is set.

    Args:
        score_model: A TargetDiff ScorePosNet3D instance (or anything with a
            .refine_net.base_block nn.ModuleList) — safe to call on any
            TargetDiff score model regardless of refine_net_type; models
            without a base_block (e.g. the 'egnn' refine net) are left
            untouched.

    Returns:
        True if checkpointing was applied to at least one layer, False if it
        was a no-op (env var unset, or no base_block to patch).
    """
    if os.environ.get("PRISM_GRAD_CKPT", "0") == "0":
        return False

    refine_net = getattr(score_model, "refine_net", None)
    base_block = getattr(refine_net, "base_block", None)
    if base_block is None:
        print("[grad_ckpt] No base_block on this refine_net (non-transformer "
              "backbone?) — nothing to patch.")
        return False

    for layer in base_block:
        _patch_layer_forward(layer)

    print(f"[grad_ckpt] Activation checkpointing enabled on "
          f"{len(base_block)} UniTransformer layer(s).")
    return True


def _patch_layer_forward(layer):
    """Rebind one layer's forward to checkpoint through the original
    implementation when training with grad enabled, else call it directly —
    same three-way gate as the original in-place change."""
    original_forward = layer.forward

    def checkpointed_forward(self, h, x, edge_attr, edge_index, mask_ligand,
                              e_w=None, fix_x=False):
        if self.training and torch.is_grad_enabled():
            return checkpoint(original_forward, h, x, edge_attr, edge_index,
                               mask_ligand, e_w, fix_x, use_reentrant=False)
        return original_forward(h, x, edge_attr, edge_index, mask_ligand,
                                 e_w=e_w, fix_x=fix_x)

    layer.forward = types.MethodType(checkpointed_forward, layer)
