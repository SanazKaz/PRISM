"""
Factory for loading a TargetDiff ScorePosNet3D checkpoint into a
TargetDiffPolicy ready to be used by PRISM's PPO loop.

Dimensions (CrossDocked, ligand_atom_mode=add_aromatic):
    protein_atom_feature_dim = 27   (6 element + 20 aa + 1 backbone)
    ligand_atom_feature_dim  = 13   (MAP_ATOM_TYPE_AROMATIC_TO_INDEX)

These are the defaults and match the published TargetDiff checkpoint.
Override via keyword args if your checkpoint used different settings.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace

import torch

# ---------------------------------------------------------------------------
# Make the vendored TargetDiff source importable
# ---------------------------------------------------------------------------
_TARGETDIFF_ROOT = Path(__file__).resolve().parents[2] / 'models' / 'targetdiff'
_TARGETDIFF_ROOT_STR = str(_TARGETDIFF_ROOT)
if _TARGETDIFF_ROOT_STR not in sys.path:
    sys.path.insert(0, _TARGETDIFF_ROOT_STR)

from models.molopt_score_model import ScorePosNet3D  # noqa: E402

from src.prism.models.targetdiff_policy import TargetDiffPolicy


# ---------------------------------------------------------------------------
# Default model config matching TargetDiff's published CrossDocked checkpoint
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_CFG = SimpleNamespace(
    model_mean_type='C0',
    beta_schedule='sigmoid',
    beta_start=1e-7,
    beta_end=2e-3,
    num_diffusion_timesteps=1000,
    loss_v_weight=100.0,
    sample_time_method='symmetric',
    v_beta_schedule='cosine',
    v_beta_s=0.01,
    pos_beta_s=0.01,          # used only if beta_schedule=='cosine'
    time_emb_dim=0,
    time_emb_mode='simple',
    center_pos_mode='protein',
    node_indicator=True,
    model_type='uni_o2',
    num_blocks=1,
    num_layers=9,
    hidden_dim=128,
    n_heads=16,
    edge_feat_dim=4,
    num_r_gaussian=20,
    knn=32,
    num_node_types=8,
    act_fn='relu',
    norm=True,
    cutoff_mode='knn',
    ew_net_type='global',
    num_x2h=1,
    num_h2x=1,
    r_max=10.0,
    x2h_out_fc=False,
    sync_twoup=False,
)


def load_targetdiff_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = 'cpu',
    protein_atom_feature_dim: int = 27,
    ligand_atom_feature_dim: int = 13,
    model_cfg: SimpleNamespace = None,
) -> TargetDiffPolicy:
    """
    Instantiate ScorePosNet3D, load weights from a TargetDiff checkpoint,
    and return a TargetDiffPolicy.

    Args:
        checkpoint_path: Path to a .pt file saved by TargetDiff's training
                         script (dict with key 'model' containing state_dict).
                         Also accepts a bare state_dict or a Lightning-style
                         checkpoint with key 'state_dict'.
        device: Target device.
        protein_atom_feature_dim: Width of protein one_hot features.
                                   27 for CrossDocked with FeaturizeProteinAtom.
        ligand_atom_feature_dim: Number of ligand atom type classes.
                                  13 for add_aromatic mode.
        model_cfg: Override the default ScorePosNet3D config.  Pass a
                   SimpleNamespace with any fields you want to change.
    Returns:
        TargetDiffPolicy (on the requested device, in eval mode).
    """
    cfg = model_cfg if model_cfg is not None else _DEFAULT_MODEL_CFG

    model = ScorePosNet3D(
        cfg,
        protein_atom_feature_dim=protein_atom_feature_dim,
        ligand_atom_feature_dim=ligand_atom_feature_dim,
    )

    checkpoint_path = Path(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle the three common checkpoint formats
    if isinstance(raw, dict):
        if 'model' in raw:
            state_dict = raw['model']           # TargetDiff native format
        elif 'state_dict' in raw:
            state_dict = raw['state_dict']      # PyTorch Lightning format
        else:
            state_dict = raw                    # bare state_dict
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {checkpoint_path}. "
            "Expected a dict with key 'model' or 'state_dict'."
        )

    # Strip any key whose tensor shape won't match the current model.
    # This happens for protein_atom_emb when the checkpoint was trained with a
    # different protein feature dim (e.g. 27) than what we instantiated (e.g. 20).
    # Those layers are intentionally re-initialised and trained from scratch.
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k in model.state_dict() and model.state_dict()[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v
    if skipped:
        print(f"[TargetDiff] Shape-mismatched keys skipped (will re-init): {skipped}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[TargetDiff] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[TargetDiff] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()

    policy = TargetDiffPolicy(
        model=model,
        num_atom_types=ligand_atom_feature_dim,
        protein_atom_feature_dim=protein_atom_feature_dim,
    )
    return policy
