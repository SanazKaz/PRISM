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
    to the width of pocket['one_hot'] produced by PRISM's data loader.
    PRISM's CrossDocked encoding uses 10-dim element-type one-hots.
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src.prism.models.base_policy import BaseDiffusionPolicy

# ---------------------------------------------------------------------------
# Make the vendored TargetDiff source importable without installing it.
# ---------------------------------------------------------------------------
_TARGETDIFF_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'targetdiff')
)
if _TARGETDIFF_ROOT not in sys.path:
    sys.path.insert(0, _TARGETDIFF_ROOT)

from models.molopt_score_model import (  # noqa: E402
    ScorePosNet3D,
    center_pos,
    extract,
    index_to_log_onehot,
    log_normal,
    log_sample_categorical,
)


class TargetDiffPolicy(BaseDiffusionPolicy):
    """
    Adapter that wraps TargetDiff's ScorePosNet3D to satisfy BaseDiffusionPolicy.

    Args:
        model: Instantiated ScorePosNet3D.
        num_atom_types: Number of ligand heavy-atom type classes (model.num_classes).
        protein_atom_feature_dim: Width of pocket['one_hot'] from the data loader.
        n_dims: Spatial dimensions (always 3).
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
    # Private helpers — shared posterior computation
    # ------------------------------------------------------------------

    def _get_posteriors(self, pred_pos, pred_v, pos_t, v_t, t, lig_mask):
        """
        Compute the reverse-diffusion posteriors for one step using
        TargetDiff's own schedule methods.

        Returns:
            pos_model_mean   [N_atoms, 3]   – posterior mean for coordinates
            pos_log_variance [N_atoms, 1]   – log variance for coordinates
            log_model_prob   [N_atoms, K]   – log posterior for atom types
        """
        # Coordinate posterior — delegate x0 recovery and mean to the model
        if self._model.model_mean_type == 'noise':
            pos0 = self._model._predict_x0_from_eps(
                xt=pos_t, eps=pred_pos - pos_t, t=t, batch=lig_mask)
        else:  # 'C0'
            pos0 = pred_pos

        pos_model_mean   = self._model.q_pos_posterior(x0=pos0, xt=pos_t, t=t, batch=lig_mask)
        pos_log_variance = extract(self._model.posterior_logvar, t, lig_mask)

        # Atom-type posterior — delegate to model's categorical diffusion methods
        log_v_recon    = F.log_softmax(pred_v, dim=-1)
        log_vt         = index_to_log_onehot(v_t, self._num_atom_types)
        log_model_prob = self._model.q_v_posterior(log_v_recon, log_vt, t, lig_mask)

        return pos_model_mean, pos_log_variance, log_model_prob

    def _step_log_prob(self, pos_s, pos_mean, pos_log_var, v_s, log_model_prob, lig_mask):
        """
        Compute per-molecule log π(z_s | z_t, pocket) for one diffusion step.

        Returns [n_mols] summed log-prob (coords + atom types).
        """
        log_p_pos = scatter_mean(
            log_normal(pos_s, pos_mean, 0.5 * pos_log_var), lig_mask, dim=0)

        v_s_onehot = F.one_hot(v_s, self._num_atom_types).float()
        log_p_v    = scatter_mean(
            (log_model_prob * v_s_onehot).sum(dim=-1), lig_mask, dim=0)

        return log_p_pos + log_p_v

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – sampling
    # ------------------------------------------------------------------

    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        """
        Run a full reverse-diffusion trajectory conditioned on the pocket.

        Mirrors ScorePosNet3D.sample_diffusion() but captures per-step
        log-probabilities and z_states needed by the PPO rollout buffer.
        Protein centering is delegated to TargetDiff's own center_pos().
        """
        device      = pocket['x'].device
        batch_protein = pocket['mask']
        protein_pos   = pocket['x'].float().clone()
        protein_v     = pocket['one_hot'].float()
        n_samples     = int(batch_protein.max().item()) + 1

        # Build ligand atom-to-molecule index
        lig_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=device), num_nodes_lig
        )

        # Centre protein using TargetDiff's own utility; init ligand at origin
        protein_pos, _, offset = center_pos(
            protein_pos,
            torch.zeros(len(lig_mask), self._n_dims, device=device),
            batch_protein, lig_mask,
            mode=self._model.center_pos_mode,
        )
        ligand_pos = torch.randn(len(lig_mask), self._n_dims, device=device)
        ligand_v   = log_sample_categorical(
            torch.zeros(len(lig_mask), self._num_atom_types, device=device)
        )

        mol_log_probs: list[torch.Tensor] = []
        z_states:      list[torch.Tensor] = []

        for i in reversed(range(self._timesteps)):
            t = torch.full((n_samples,), i, dtype=torch.long, device=device)

            preds = self._model(
                protein_pos=protein_pos, protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos, init_ligand_v=ligand_v,
                batch_ligand=lig_mask, time_step=t,
            )

            pos_mean, pos_log_var, log_model_prob = self._get_posteriors(
                preds['pred_ligand_pos'], preds['pred_ligand_v'],
                ligand_pos, ligand_v, t, lig_mask,
            )

            # Sample next state — no noise injected at t == 0
            nonzero = (1 - (t == 0).float())[lig_mask].unsqueeze(-1)
            ligand_pos = pos_mean + nonzero * (0.5 * pos_log_var).exp() * torch.randn_like(ligand_pos)
            ligand_v   = log_sample_categorical(log_model_prob)

            mol_log_probs.append(
                self._step_log_prob(ligand_pos, pos_mean, pos_log_var, ligand_v, log_model_prob, lig_mask)
            )
            z_states.append(
                torch.cat([ligand_pos, F.one_hot(ligand_v, self._num_atom_types).float()], dim=-1)
            )

        # Restore absolute coordinates
        ligand_pos  = ligand_pos  + offset[lig_mask]
        protein_pos = protein_pos + offset[batch_protein]

        xh_lig    = torch.cat([ligand_pos, F.one_hot(ligand_v, self._num_atom_types).float()], dim=-1)
        xh_pocket = torch.cat([protein_pos, protein_v], dim=-1)

        return xh_lig, xh_pocket, lig_mask, batch_protein, mol_log_probs, z_states

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – log-probability (PPO update)
    # ------------------------------------------------------------------

    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """
        Compute log π_θ(z_s | z_t, pocket) with the *current* model weights.

        z_t / z_s: [N_atoms, 3 + num_atom_types]
        xh_pock:   [N_pock,  3 + protein_nf]

        Returns [n_mols] per-molecule log-probability.
        """
        t_int = (t.squeeze(-1) * self._timesteps).round().long()

        pos_t = z_t[:, :self._n_dims];  v_t = z_t[:, self._n_dims:].argmax(dim=-1)
        pos_s = z_s[:, :self._n_dims];  v_s = z_s[:, self._n_dims:].argmax(dim=-1)

        preds = self._model(
            protein_pos=xh_pock[:, :self._n_dims].float(),
            protein_v=xh_pock[:, self._n_dims:].float(),
            batch_protein=poc_mask,
            init_ligand_pos=pos_t, init_ligand_v=v_t,
            batch_ligand=lig_mask, time_step=t_int,
        )

        pos_mean, pos_log_var, log_model_prob = self._get_posteriors(
            preds['pred_ligand_pos'], preds['pred_ligand_v'],
            pos_t, v_t, t_int, lig_mask,
        )

        return self._step_log_prob(pos_s, pos_mean, pos_log_var, v_s, log_model_prob, lig_mask)

    def log_p_zs_given_zt_channels(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """Diagnostic-only variant of log_p_zs_given_zt.

        Returns the two additive channels of the policy log-prob *separately*:

            (log_p_pos, log_p_v)   each shape [n_mols]

        so callers can compare their magnitudes and (after backward on each)
        the gradient each channel sends into the trainable params. Used to test
        whether the continuous-coordinate Gaussian term swamps the categorical
        atom-type term — which would starve atom-identity rewards (QED/logP/SA).
        """
        t_int = (t.squeeze(-1) * self._timesteps).round().long()

        pos_t = z_t[:, :self._n_dims];  v_t = z_t[:, self._n_dims:].argmax(dim=-1)
        pos_s = z_s[:, :self._n_dims];  v_s = z_s[:, self._n_dims:].argmax(dim=-1)

        preds = self._model(
            protein_pos=xh_pock[:, :self._n_dims].float(),
            protein_v=xh_pock[:, self._n_dims:].float(),
            batch_protein=poc_mask,
            init_ligand_pos=pos_t, init_ligand_v=v_t,
            batch_ligand=lig_mask, time_step=t_int,
        )

        pos_mean, pos_log_var, log_model_prob = self._get_posteriors(
            preds['pred_ligand_pos'], preds['pred_ligand_v'],
            pos_t, v_t, t_int, lig_mask,
        )

        log_p_pos = scatter_mean(
            log_normal(pos_s, pos_mean, 0.5 * pos_log_var), lig_mask, dim=0)
        v_s_onehot = F.one_hot(v_s, self._num_atom_types).float()
        log_p_v = scatter_mean(
            (log_model_prob * v_s_onehot).sum(dim=-1), lig_mask, dim=0)
        return log_p_pos, log_p_v

    def trajectory_stats_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """Diagnostic-only: per-molecule signals for the denoising-trajectory sweep.

        One forward pass; returns a dict of [n_mols] tensors:

            log_p_pos, log_p_v   – the two additive policy log-prob channels
            pos_sigma            – posterior coordinate std σ(t); the 1/σ² that
                                   blows up log_p_pos at low noise. Also a proxy
                                   for coordinate malleability (→0 ⇒ coords frozen)
            atomtype_entropy     – entropy of the categorical atom-type posterior
                                   (atom-identity malleability; →0 once an atom's
                                   element is locked in)

        Lets the sweep show WHEN along the reverse trajectory the molecule's
        identity and geometry crystallize, and where the channel gradient is healthy.
        """
        t_int = (t.squeeze(-1) * self._timesteps).round().long()

        pos_t = z_t[:, :self._n_dims];  v_t = z_t[:, self._n_dims:].argmax(dim=-1)
        pos_s = z_s[:, :self._n_dims];  v_s = z_s[:, self._n_dims:].argmax(dim=-1)

        preds = self._model(
            protein_pos=xh_pock[:, :self._n_dims].float(),
            protein_v=xh_pock[:, self._n_dims:].float(),
            batch_protein=poc_mask,
            init_ligand_pos=pos_t, init_ligand_v=v_t,
            batch_ligand=lig_mask, time_step=t_int,
        )

        pos_mean, pos_log_var, log_model_prob = self._get_posteriors(
            preds['pred_ligand_pos'], preds['pred_ligand_v'],
            pos_t, v_t, t_int, lig_mask,
        )

        log_p_pos = scatter_mean(
            log_normal(pos_s, pos_mean, 0.5 * pos_log_var), lig_mask, dim=0)
        v_s_onehot = F.one_hot(v_s, self._num_atom_types).float()
        log_p_v = scatter_mean(
            (log_model_prob * v_s_onehot).sum(dim=-1), lig_mask, dim=0)

        pos_sigma = scatter_mean(
            (0.5 * pos_log_var).exp().squeeze(-1), lig_mask, dim=0)
        p = log_model_prob.exp()
        ent_atom = -(p * log_model_prob).sum(dim=-1)        # per-atom entropy
        atomtype_entropy = scatter_mean(ent_atom, lig_mask, dim=0)

        return {
            'log_p_pos': log_p_pos, 'log_p_v': log_p_v,
            'pos_sigma': pos_sigma, 'atomtype_entropy': atomtype_entropy,
        }

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – data pre-processing
    # ------------------------------------------------------------------

    def get_ligand_and_pocket(self, data):
        """Convert a raw PRISM dataloader batch into (ligand_dict, pocket_dict, names)."""
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
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self
