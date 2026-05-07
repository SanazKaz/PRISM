"""
TargetDiff integration for PRISM.

TargetDiff paper: https://arxiv.org/abs/2303.03543
Source:           src/models/targetdiff/  (vendored)

Design notes
------------
Latent representation
    PRISM's rollout buffer stores z_states as [N_atoms, D] tensors.  For
    TargetDiff we pack D = n_dims + num_atom_types and use the layout:

        z[:, :3]           – 3-D ligand coordinates (continuous)
        z[:, 3:]           – atom-type one-hot  (float, stored after argmax)

    During log_p_zs_given_zt we reconstruct integer indices with argmax so
    the discrete q_v_posterior can be evaluated.

Timestep convention
    PRISM normalises timesteps to [0, 1].  TargetDiff uses integer indices
    [0, T-1].  We convert with  t_int = round(t_norm * T).

time_emb_mode
    Only 'simple' mode is supported.  The 'sin' branch in TargetDiff's
    forward() is missing a [batch_ligand] index and would produce
    wrong-shaped tensors.  Use  time_emb_mode: simple  in your config.

Pocket feature dimensions
    ScorePosNet3D must be instantiated with protein_atom_feature_dim equal
    to the width of pocket['one_hot'] produced by PRISM's data loader.  The
    PRISM CrossDocked pocket encoding is 27-dimensional; set this in the
    TargetDiff config accordingly.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean

from src.prism.models.base_policy import BaseDiffusionPolicy

# ---------------------------------------------------------------------------
# Make the vendored TargetDiff source importable without installing it.
# We insert its root on sys.path so that `from models.xxx import ...` works.
# ---------------------------------------------------------------------------
_TARGETDIFF_ROOT = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'models', 'targetdiff'
)
_TARGETDIFF_ROOT = os.path.normpath(_TARGETDIFF_ROOT)
if _TARGETDIFF_ROOT not in sys.path:
    sys.path.insert(0, _TARGETDIFF_ROOT)

from models.molopt_score_model import (  # noqa: E402  (after sys.path insert)
    ScorePosNet3D,
    index_to_log_onehot,
    log_sample_categorical,
    log_normal,
    extract,
    center_pos,
)


class TargetDiffPolicy(BaseDiffusionPolicy):
    """
    Adapter that wraps TargetDiff's ScorePosNet3D to satisfy BaseDiffusionPolicy.

    Args:
        model (ScorePosNet3D): Instantiated TargetDiff score model.
        num_atom_types (int): Number of ligand heavy-atom type classes.
            Must match model.num_classes.
        protein_atom_feature_dim (int): Width of pocket['one_hot'] from the
            PRISM data loader.  Must match model's protein_atom_emb input dim.
        n_dims (int): Spatial dimensions (always 3).
    """

    def __init__(
        self,
        model: ScorePosNet3D,
        num_atom_types: int,
        protein_atom_feature_dim: int,
        n_dims: int = 3,
    ):
        super().__init__()
        self._model = model
        self._num_atom_types = num_atom_types
        self._protein_atom_feature_dim = protein_atom_feature_dim
        self._n_dims = n_dims
        self._timesteps = model.num_timesteps

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – sampling
    # ------------------------------------------------------------------

    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        """
        Run a full reverse-diffusion trajectory conditioned on the pocket and
        return the standard PRISM 6-tuple.

        The sampling loop mirrors ScorePosNet3D.sample_diffusion() but runs
        without torch.no_grad so gradients can be re-computed during the PPO
        update, and captures per-step log-probabilities.

        Latent z is stored as cat([pos, one_hot(v).float()], dim=-1).
        """
        device = pocket['x'].device
        batch_protein = pocket['mask']           # [N_prot_atoms]  molecule idx
        protein_pos   = pocket['x'].clone()      # [N_prot_atoms, 3]
        protein_v     = pocket['one_hot']        # [N_prot_atoms, prot_nf]
        n_samples     = int(batch_protein.max().item()) + 1

        # Centre protein; keep offset to restore coordinates afterwards
        offset = scatter_mean(protein_pos, batch_protein, dim=0)   # [n_mols, 3]
        protein_pos = protein_pos - offset[batch_protein]

        # Build ligand atom-to-molecule index
        lig_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=device), num_nodes_lig
        )  # [N_lig_atoms]

        # Initialise from prior: positions near pocket centre, uniform types
        pocket_centres = scatter_mean(protein_pos, batch_protein, dim=0)  # [n_mols, 3]
        init_ligand_pos = pocket_centres[lig_mask] + torch.randn(
            len(lig_mask), self._n_dims, device=device
        )
        uniform_logits = torch.zeros(len(lig_mask), self._num_atom_types, device=device)
        init_ligand_v  = log_sample_categorical(uniform_logits)   # [N_lig_atoms] int

        ligand_pos = init_ligand_pos
        ligand_v   = init_ligand_v

        mol_log_probs: list[torch.Tensor] = []
        z_states:      list[torch.Tensor] = []

        # Reverse diffusion: t goes from T-1 down to 0
        time_seq = list(reversed(range(self._timesteps)))
        for i in time_seq:
            t = torch.full((n_samples,), i, dtype=torch.long, device=device)

            preds = self._model(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=lig_mask,
                time_step=t,
            )
            pred_ligand_pos = preds['pred_ligand_pos']
            pred_ligand_v   = preds['pred_ligand_v']

            # ---- coordinate posterior ----
            if self._model.model_mean_type == 'noise':
                pred_pos_noise = pred_ligand_pos - ligand_pos
                pos0 = self._model._predict_x0_from_eps(
                    xt=ligand_pos, eps=pred_pos_noise, t=t, batch=lig_mask
                )
            else:  # 'C0'
                pos0 = pred_ligand_pos

            pos_model_mean  = self._model.q_pos_posterior(
                x0=pos0, xt=ligand_pos, t=t, batch=lig_mask
            )  # [N_lig_atoms, 3]
            pos_log_variance = extract(self._model.posterior_logvar, t, lig_mask)
            # no noise at t == 0
            nonzero_mask = (1 - (t == 0).float())[lig_mask].unsqueeze(-1)
            ligand_pos_next = (
                pos_model_mean
                + nonzero_mask * (0.5 * pos_log_variance).exp()
                * torch.randn_like(ligand_pos)
            )

            # ---- atom-type posterior ----
            log_v_recon   = F.log_softmax(pred_ligand_v, dim=-1)
            log_vt        = index_to_log_onehot(ligand_v, self._num_atom_types)
            log_model_prob = self._model.q_v_posterior(log_v_recon, log_vt, t, lig_mask)
            ligand_v_next  = log_sample_categorical(log_model_prob)

            # ---- per-step log π(z_s | z_t) of the SAMPLED next state ----
            log_prob_pos = log_normal(
                ligand_pos_next, pos_model_mean, 0.5 * pos_log_variance
            )  # [N_lig_atoms]
            log_prob_pos_mol = scatter_mean(log_prob_pos, lig_mask, dim=0)  # [n_mols]

            v_next_onehot = F.one_hot(ligand_v_next, self._num_atom_types).float()
            log_prob_v_atom = (log_model_prob * v_next_onehot).sum(dim=-1)
            log_prob_v_mol  = scatter_mean(log_prob_v_atom, lig_mask, dim=0)

            mol_log_probs.append(log_prob_pos_mol + log_prob_v_mol)

            ligand_pos = ligand_pos_next
            ligand_v   = ligand_v_next

            z_states.append(torch.cat(
                [ligand_pos, F.one_hot(ligand_v, self._num_atom_types).float()],
                dim=-1,
            ))

        # Restore absolute coordinates
        ligand_pos  = ligand_pos  + offset[lig_mask]
        protein_pos = protein_pos + offset[batch_protein]

        xh_lig    = torch.cat(
            [ligand_pos, F.one_hot(ligand_v, self._num_atom_types).float()], dim=-1
        )
        xh_pocket = torch.cat([protein_pos, protein_v], dim=-1)

        return xh_lig, xh_pocket, lig_mask, batch_protein, mol_log_probs, z_states

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – log-probability (PPO update)
    # ------------------------------------------------------------------

    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """
        Compute log π_θ(z_s | z_t, pocket) with the *current* model weights.

        z_t / z_s layout:  [N_atoms, 3 + num_atom_types]
        xh_pock layout:    [N_pock,  3 + protein_nf]

        Returns [n_mols] per-molecule log-probability.
        """
        # Convert normalised timestep → integer index
        t_int = (t.squeeze(-1) * self._timesteps).round().long()   # [n_mols]

        # Unpack latents
        pos_t = z_t[:, :self._n_dims]                              # [N_atoms, 3]
        v_t   = z_t[:, self._n_dims:].argmax(dim=-1)              # [N_atoms] int
        pos_s = z_s[:, :self._n_dims]
        v_s   = z_s[:, self._n_dims:].argmax(dim=-1)

        # Unpack pocket
        protein_pos = xh_pock[:, :self._n_dims]
        protein_v   = xh_pock[:, self._n_dims:]

        # Forward pass with current weights
        preds = self._model(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=poc_mask,
            init_ligand_pos=pos_t,
            init_ligand_v=v_t,
            batch_ligand=lig_mask,
            time_step=t_int,
        )
        pred_ligand_pos = preds['pred_ligand_pos']
        pred_ligand_v   = preds['pred_ligand_v']

        # ---- coordinate log-prob ----
        if self._model.model_mean_type == 'noise':
            pred_pos_noise = pred_ligand_pos - pos_t
            pos0 = self._model._predict_x0_from_eps(
                xt=pos_t, eps=pred_pos_noise, t=t_int, batch=lig_mask
            )
        else:
            pos0 = pred_ligand_pos

        pos_model_mean   = self._model.q_pos_posterior(
            x0=pos0, xt=pos_t, t=t_int, batch=lig_mask
        )
        pos_log_variance = extract(self._model.posterior_logvar, t_int, lig_mask)

        log_prob_pos     = log_normal(pos_s, pos_model_mean, 0.5 * pos_log_variance)
        log_prob_pos_mol = scatter_mean(log_prob_pos, lig_mask, dim=0)

        # ---- atom-type log-prob ----
        log_v_recon    = F.log_softmax(pred_ligand_v, dim=-1)
        log_vt         = index_to_log_onehot(v_t, self._num_atom_types)
        log_model_prob = self._model.q_v_posterior(log_v_recon, log_vt, t_int, lig_mask)

        v_s_onehot      = F.one_hot(v_s, self._num_atom_types).float()
        log_prob_v_atom = (log_model_prob * v_s_onehot).sum(dim=-1)
        log_prob_v_mol  = scatter_mean(log_prob_v_atom, lig_mask, dim=0)

        return log_prob_pos_mol + log_prob_v_mol

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – data pre-processing
    # ------------------------------------------------------------------

    def get_ligand_and_pocket(self, data):
        """
        Convert a raw PRISM dataloader batch into (ligand_dict, pocket_dict, names).

        PRISM's CrossDocked data loader already produces pocket tensors with
        keys 'x' (positions) and 'one_hot' (features).  TargetDiff's
        ScorePosNet3D consumes those directly via protein_atom_emb, provided
        the model is initialised with protein_atom_feature_dim matching the
        feature width.  No additional transformation is needed here.

        If the raw batch format differs from the DiffSBDD convention, override
        this method in a subclass.
        """
        pocket = {
            'x':       data['pocket_coords'],
            'one_hot': data['pocket_one_hot'],
            'size':    data['num_pocket_nodes'],
            'mask':    data['pocket_mask'],
        }
        ligand = {
            'x':       data.get('ligand_coords', None),
            'one_hot': data.get('ligand_one_hot', None),
            'size':    data.get('num_ligand_nodes', None),
            'mask':    data.get('ligand_mask', None),
        }
        names = data.get('names', [None] * int(data['num_pocket_nodes'].shape[0]))
        return ligand, pocket, names

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – required attributes
    # ------------------------------------------------------------------

    @property
    def atom_nf(self) -> int:
        return self._num_atom_types

    @property
    def n_dims(self) -> int:
        return self._n_dims

    # ------------------------------------------------------------------
    # nn.Module routing – only ScorePosNet3D params are trained
    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self._model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self
