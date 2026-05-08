# src/prism/models/policy_factory.py
"""
Factory functions for constructing PRISM policy objects from config.

Keeping model-construction logic here (rather than in lightning_module.py)
means the Lightning module stays a thin orchestrator, and all knowledge about
checkpoint formats, architecture defaults, and dataset encodings lives in one
place alongside the model code.

Public API
----------
build_diffsbdd_policy(config, device, node_histogram, warm_start_checkpoint)
    -> (DiffSBDDPolicy, LigandPocketDDPM, dataset_info)

build_targetdiff_policy(config, device, warm_start_checkpoint)
    -> (TargetDiffPolicy, dataset_info)

load_diffsbdd_weights(ddpm_module, checkpoint_path)
    Robust weight loading that handles every DiffSBDD checkpoint format.
"""

import torch
from argparse import Namespace
from pathlib import Path

from src.models.diffsbdd.lightning_modules import LigandPocketDDPM
from src.prism.models.diffsbdd_policy import DiffSBDDPolicy
from src.prism.models.targetdiff_factory import load_targetdiff_policy


# ---------------------------------------------------------------------------
# DiffSBDD architecture defaults (CrossDocked checkpoint geometry).
#
# These are injected whenever the corresponding section is absent from the
# config, so standard CrossDocked runs need no egnn_params / diffusion_params
# block at all.  Non-standard checkpoints (e.g. BindingMOAD) place overrides
# under config.model.egnn_params / config.model.diffusion_params, which are
# merged on top of these defaults at build time.
# ---------------------------------------------------------------------------

_DIFFSBDD_EGNN_DEFAULTS = {
    # Spatial cutoffs (Ångström) for building the interaction graph.
    # None means no cutoff on ligand-internal edges (they're always connected).
    'device': 'cuda',
    'edge_cutoff_ligand': None,
    'edge_cutoff_pocket': 5.0,
    'edge_cutoff_interaction': 5.0,
    'reflection_equivariant': False,
    # Architecture dimensions — must match the pretrained checkpoint.
    'joint_nf': 32,        # joint node feature dimension
    'hidden_nf': 128,      # EGNN hidden dimension
    'n_layers': 5,         # number of EGNN message-passing layers
    'attention': True,
    'tanh': True,
    'norm_constant': 1,
    'inv_sublayers': 1,
    'sin_embedding': False,
    'aggregation_method': 'sum',
    'normalization_factor': 100,
}

_DIFFSBDD_DIFFUSION_DEFAULTS = {
    # Noise schedule shape — polynomial_2 gives a smooth variance curve that
    # avoids the sharp boundaries of cosine/linear schedules.
    'diffusion_noise_schedule': 'polynomial_2',
    # Small epsilon added to the schedule to keep SNR bounded.
    'diffusion_noise_precision': 5.0e-4,
    'diffusion_loss_type': 'l2',
    # Normalisation applied to coordinates and atom-type one-hots before
    # the diffusion forward process.  [coord_scale, onehot_scale].
    'normalize_factors': [1, 4],
    # Note: 'diffusion_steps' is NOT here — it comes from config.model.total_timesteps
    # so that a single config knob controls both the rollout collector and DiffSBDD.
}

_DIFFSBDD_TRAINING_DEFAULTS = {
    # Task mode — always pocket-conditioned generation for PRISM.
    'mode': 'pocket_conditioning',
    # Use full heavy-atom pocket representation (not C-alpha only).
    'pocket_representation': 'full-atom',
    # Data augmentation — off during PPO fine-tuning.
    'augment_noise': 0,
    'augment_rotation': False,
    # Gradient clipping is handled by PPO's own max_grad_norm, but DiffSBDD's
    # Lightning module expects the flag to exist.
    'clip_grad': True,
    # Auxiliary geometric loss from the original DiffSBDD training — disabled
    # during PPO because rewards replace it.
    'auxiliary_loss': False,
    'loss_params': {'max_weight': 0.001, 'schedule': 'linear', 'clamp_lj': 3.0},
}


# ---------------------------------------------------------------------------
# TargetDiff atom-type decoder for the 'add_aromatic' encoding (13 classes).
#
# TargetDiff's MAP_ATOM_TYPE_AROMATIC_TO_INDEX (transforms.py) encodes each
# heavy atom as (atomic_number, is_aromatic).  The 13 entries in index order:
#
#   0  (1,  False)  H
#   1  (6,  False)  C   non-aromatic
#   2  (6,  True)   C   aromatic
#   3  (7,  False)  N   non-aromatic
#   4  (7,  True)   N   aromatic
#   5  (8,  False)  O   non-aromatic
#   6  (8,  True)   O   aromatic
#   7  (9,  False)  F
#   8  (15, False)  P   non-aromatic
#   9  (15, True)   P   aromatic
#  10  (16, False)  S   non-aromatic
#  11  (16, True)   S   aromatic
#  12  (17, False)  Cl
#
# Duplicates (C/C, N/N, …) are intentional: both non-aromatic and aromatic
# variants of the same element decode to the same element symbol so that
# RDKit can reconstruct the molecule and infer aromaticity itself.
# The colors_dic and radius_dic are taken directly from the DiffSBDD
# 'crossdock' dataset_params so visualisation is consistent.
# ---------------------------------------------------------------------------

_TARGETDIFF_ATOM_DECODER = ['H', 'C', 'C', 'N', 'N', 'O', 'O', 'F', 'P', 'P', 'S', 'S', 'Cl']

_TARGETDIFF_DATASET_INFO = {
    'atom_decoder': _TARGETDIFF_ATOM_DECODER,
    # atom_encoder maps element symbol -> last index with that symbol.
    # Used only for inverse lookups (e.g. encoding reference ligands).
    # Where duplicates exist (C, N, O, P, S each appear twice for non-/aromatic
    # variants), the higher index wins — that's fine because PRISM only uses the
    # decoder direction at inference time.
    'atom_encoder': {sym: i for i, sym in enumerate(_TARGETDIFF_ATOM_DECODER)},
    # One colour and radius entry per class (13 total), mirroring the PyMOL
    # palette used by DiffSBDD's crossdock dataset_params in constants.py.
    # Aromatic variants of an element share the same colour as the non-aromatic.
    'colors_dic': [
        '#ffffff',          # 0  H
        '#33ff33', '#33ff33',  # 1  C,  2  C(ar)
        '#3333ff', '#3333ff',  # 3  N,  4  N(ar)
        '#ff4d4d', '#ff4d4d',  # 5  O,  6  O(ar)
        '#B3FFFF',          # 7  F
        '#ff8000', '#ff8000',  # 8  P,  9  P(ar)
        '#e6c540', '#e6c540',  # 10 S,  11 S(ar)
        '#1FF01F',          # 12 Cl
    ],
    'radius_dic': [0.3] * 13,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _ns_to_dict(val):
    """Convert an argparse Namespace (or any dict-like) to a plain dict.

    Used when reading optional override sections from the YAML config, which
    dict_to_namespace() has already converted to Namespace objects.
    Returns an empty dict when val is None so callers can always unpack safely.
    """
    if val is None:
        return {}
    return vars(val) if hasattr(val, '__dict__') else dict(val)


# ---------------------------------------------------------------------------
# DiffSBDD
# ---------------------------------------------------------------------------

def load_diffsbdd_weights(ddpm_module, checkpoint_path):
    """Load DiffSBDD model weights from any PRISM or upstream checkpoint format.

    Handles three common layouts:
    - PRISM .pt (direct save)   : bare state_dict, no key prefix.
    - PRISM .ckpt (Lightning)   : state_dict nested under 'state_dict' key,
                                  with keys prefixed by 'ddpm_model.' or
                                  'policy._ddpm_module.' depending on PRISM version.
    - DiffSBDD upstream .pt     : bare state_dict, no prefix.

    Optimizer state in .ckpt files is intentionally ignored so a fresh
    optimizer is built for the new freeze configuration.

    Args:
        ddpm_module:      The LigandPocketDDPM instance to load into.
        checkpoint_path:  Path to the checkpoint file.
    """
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        # Lightning .ckpt — the full model state lives under 'state_dict',
        # with a module-level prefix that depends on which PRISM version saved it.
        full_sd = raw['state_dict']
        for prefix in ('ddpm_model.', 'policy._ddpm_module.'):
            stripped = {k[len(prefix):]: v
                        for k, v in full_sd.items() if k.startswith(prefix)}
            if stripped:
                missing, unexpected = ddpm_module.load_state_dict(stripped, strict=False)
                print(f"[WarmStart] Loaded from Lightning ckpt "
                      f"(prefix='{prefix}', {len(stripped)} keys, "
                      f"{len(missing)} missing, {len(unexpected)} unexpected)")
                return
        # Fallback: no recognised prefix found — try the dict as-is.
        # This handles edge cases where the ckpt was already stripped.
        missing, unexpected = ddpm_module.load_state_dict(full_sd, strict=False)
        print(f"[WarmStart] Loaded from state_dict (no prefix stripped, "
              f"{len(missing)} missing, {len(unexpected)} unexpected)")
    else:
        # Direct .pt file — either PRISM-saved or original DiffSBDD.
        state_dict = raw if isinstance(raw, dict) else {}
        missing, unexpected = ddpm_module.load_state_dict(state_dict, strict=False)
        print(f"[WarmStart] Loaded from .pt "
              f"({len(state_dict)} keys, {len(missing)} missing, "
              f"{len(unexpected)} unexpected)")


def build_diffsbdd_policy(config, device, node_histogram, warm_start_checkpoint):
    """Instantiate a DiffSBDD policy from config, with hardcoded CrossDocked defaults.

    Architecture and training-mode parameters (egnn_params, diffusion_params,
    mode, etc.) are never required in the YAML config.  Sensible CrossDocked
    defaults are injected automatically.  Non-standard checkpoints (BindingMOAD)
    can override individual values via config.model.egnn_params and
    config.model.diffusion_params, which are merged on top of the defaults.

    Args:
        config:                 Parsed YAML config Namespace.
        device:                 torch.device to move the model to.
        node_histogram:         Per-dataset atom-count distribution (list),
                                used by LigandPocketDDPM for conditional sizing.
        warm_start_checkpoint:  Optional path to load weights from before training.

    Returns:
        policy      (DiffSBDDPolicy):      Wrapped model satisfying BaseDiffusionPolicy.
        ddpm_module (LigandPocketDDPM):    Raw DiffSBDD module (needed for validation).
        dataset_info (dict):               Atom encoder/decoder + bond tables.
    """
    # Keys that are PRISM-only and must not be forwarded to LigandPocketDDPM.
    # ppo_params is kept for legacy compatibility with any old checkpoints that
    # might still have it present at the top level.
    _EXCLUDE = {
        'ppo', 'ppo_params',
        'enable_progress_bar', 'num_sanity_val_steps', 'wandb_params',
        'gpus', 'n_epochs', 'logdir', 'fp16', 'run_identifier',
        'reward_params', 'docking_params', 'model_type',
        'freeze_except', 'model',
    }
    ddpm_config = {k: v for k, v in vars(config).items() if k not in _EXCLUDE}

    # config.model is optional — standard CrossDocked runs omit it entirely.
    model_cfg = getattr(config, 'model', None)

    # --- lr: LigandPocketDDPM requires lr even though PRISM replaces its optimizer.
    # Use the PPO learning rate as a sensible stand-in.
    if 'lr' not in ddpm_config:
        ddpm_config['lr'] = config.ppo.lr

    # --- EGNN architecture ---
    # Merge CrossDocked defaults with any per-config overrides.  The override
    # dict is empty for standard configs, so defaults pass through unchanged.
    # LigandPocketDDPM type-hints egnn_params as Namespace, so convert.
    if 'egnn_params' not in ddpm_config:
        override = _ns_to_dict(getattr(model_cfg, 'egnn_params', None))
        ddpm_config['egnn_params'] = Namespace(**{**_DIFFSBDD_EGNN_DEFAULTS, **override})

    # --- Diffusion schedule ---
    # config.model.total_timesteps is the single source of truth for the chain
    # length; it drives both diffusion_steps here and the rollout collector.
    # LigandPocketDDPM accesses diffusion_params as attributes, so use Namespace.
    if 'diffusion_params' not in ddpm_config:
        total_ts = getattr(model_cfg, 'total_timesteps', 500) if model_cfg else 500
        override = _ns_to_dict(getattr(model_cfg, 'diffusion_params', None))
        ddpm_config['diffusion_params'] = Namespace(**{
            'diffusion_steps': total_ts,
            **_DIFFSBDD_DIFFUSION_DEFAULTS,
            **override,   # e.g. BindingMOAD changes diffusion_noise_precision
        })

    # --- Training-mode flags ---
    # Only inject a default if the key is entirely absent from the config
    # (preserves any top-level overrides a user may have kept in their YAML).
    for key, val in _DIFFSBDD_TRAINING_DEFAULTS.items():
        if key not in ddpm_config:
            ddpm_config[key] = val

    ddpm_module = LigandPocketDDPM(
        outdir=Path(config.logdir),
        node_histogram=node_histogram,
        **ddpm_config,
    )
    ddpm_module.to(device)

    if warm_start_checkpoint is not None:
        load_diffsbdd_weights(ddpm_module, warm_start_checkpoint)

    policy = DiffSBDDPolicy(ddpm_module)
    # dataset_info comes from the model itself (populated from constants.py
    # during LigandPocketDDPM.__init__ based on the dataset key in config).
    dataset_info = ddpm_module.dataset_info.copy()
    return policy, ddpm_module, dataset_info


# ---------------------------------------------------------------------------
# TargetDiff
# ---------------------------------------------------------------------------

def build_targetdiff_policy(config, device, warm_start_checkpoint):
    """Instantiate a TargetDiff policy from config.

    Resolves the checkpoint path and feature dimensions from config.model.*,
    with fallback to the legacy top-level keys (targetdiff_checkpoint, etc.)
    for backwards compatibility with old config files.

    The dataset_info returned uses the correct 13-class 'add_aromatic' decoder
    derived from TargetDiff's MAP_ATOM_TYPE_AROMATIC_TO_INDEX (transforms.py).
    Colors and radii are reused from the DiffSBDD CrossDocked dataset_params.

    Args:
        config:                 Parsed YAML config Namespace.
        device:                 torch.device to move the model to.
        warm_start_checkpoint:  Fallback checkpoint path if config.model.checkpoint
                                is not set (rare; usually only used in tests).

    Returns:
        policy      (TargetDiffPolicy):  Wrapped model satisfying BaseDiffusionPolicy.
        dataset_info (dict):             Atom encoder/decoder for reward scoring.
    """
    model_cfg = getattr(config, 'model', None)

    # Checkpoint path: new config key takes priority, then legacy key, then CLI arg.
    checkpoint_path = (
        getattr(model_cfg, 'checkpoint', None) if model_cfg else None
    ) or getattr(config, 'targetdiff_checkpoint', None) or warm_start_checkpoint

    if checkpoint_path is None:
        raise ValueError(
            "TargetDiff requires a pretrained checkpoint. "
            "Set config.model.checkpoint in your YAML, "
            "or pass --warm_start_from_ddpm on the command line."
        )

    # Protein feature dimension: PRISM CrossDocked encodes pocket atoms as
    # 10-dim element-type one-hots (C,N,O,S,B,Br,Cl,P,I,F).  TargetDiff's
    # pretrained protein_atom_emb layer was trained on 27-dim features, so
    # that layer is re-initialised when there is a shape mismatch on load.
    protein_feat_dim = (
        getattr(model_cfg, 'protein_feat_dim', None) if model_cfg else None
    ) or getattr(config, 'targetdiff_protein_feat_dim', 27)

    # Number of ligand atom classes — 13 for the 'add_aromatic' CrossDocked encoding.
    ligand_atom_types = (
        getattr(model_cfg, 'ligand_atom_types', None) if model_cfg else None
    ) or getattr(config, 'targetdiff_ligand_atom_types', 13)

    policy = load_targetdiff_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        protein_atom_feature_dim=protein_feat_dim,
        ligand_atom_feature_dim=ligand_atom_types,
    )

    # Return a copy so callers can safely mutate it (e.g. add 'datadir').
    dataset_info = {k: list(v) if isinstance(v, list) else v
                    for k, v in _TARGETDIFF_DATASET_INFO.items()}
    return policy, dataset_info
