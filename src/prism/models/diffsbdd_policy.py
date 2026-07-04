import torch
import torch.nn as nn
from pathlib import Path
from torch_scatter import scatter_mean

from src.prism.models.base_policy import BaseDiffusionPolicy
from src.models.diffsbdd.lightning_modules import LigandPocketDDPM


class DiffSBDDPolicy(BaseDiffusionPolicy):
    """
    Thin adapter that wraps LigandPocketDDPM so the existing DiffSBDD model
    satisfies the BaseDiffusionPolicy interface expected by the PPO loop.

    The outer LigandPocketDDPM handles data pre-processing and validation;
    the inner ConditionalDDPM (self._ddpm_module.ddpm) owns the generative
    parameters that are actually trained by PPO.

    gradient / parameter calls on this object are forwarded to the inner
    ConditionalDDPM only, matching the original behaviour where
    PPOAlgorithm used policy_network.ddpm directly.
    """

    def __init__(self, ddpm_module: LigandPocketDDPM):
        super().__init__()
        self._ddpm_module = ddpm_module
        # Register the inner model so PyTorch tracks its parameters correctly.
        self._inner = ddpm_module.ddpm

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – sampling
    # ------------------------------------------------------------------

    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        return self._inner.sample_given_pocket(pocket, num_nodes_lig, **kwargs)

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – log-probability
    # ------------------------------------------------------------------

    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        return self._inner.log_p_zs_given_zt(s, t, z_t, z_s, xh_pock, lig_mask, poc_mask)

    def log_p_zs_given_zt_channels(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """Returns (log_p_pos, log_p_v) — Gaussian coord and atom-type channels separately.

        Both DiffSBDD channels are continuous Gaussian; we split by dimension.
        Implemented here (not on the vendored ConditionalDDPM) so the vendor tree
        stays unmodified. NOTE: `sigma` is a per-atom scalar of shape [N, 1] (from
        inflate_batch_array — molecule-axis only, NOT per-dimension), so it must not
        be sliced by channel; broadcast it against each channel's `diff` slice and
        scale the log-normalization term by the channel's dimension count. The
        quadratic terms split exactly as n_dims + atom_nf.
        """
        inner = self._inner
        gamma_s = inner.gamma(s)
        gamma_t = inner.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            inner.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = inner.sigma(gamma_s, target_tensor=z_t)
        sigma_t = inner.sigma(gamma_t, target_tensor=z_t)

        eps_t_lig, _ = inner.dynamics(z_t, xh_pock, t, lig_mask, poc_mask)

        mu_lig = z_t / alpha_t_given_s[lig_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[lig_mask] * eps_t_lig

        sigma = sigma_t_given_s * sigma_s / sigma_t
        sigma_sq_lig = sigma[lig_mask] ** 2 + 1e-8              # [N, 1] broadcast scalar
        log_2pi_sig = torch.log(2 * torch.pi * sigma_sq_lig).squeeze(-1)   # [N]
        diff = z_s - mu_lig

        diff_pos = diff[:, :inner.n_dims]
        log_p_pos_atom = -0.5 * (
            (diff_pos ** 2 / sigma_sq_lig).sum(dim=1) + diff_pos.shape[1] * log_2pi_sig
        )
        diff_v = diff[:, inner.n_dims:]
        log_p_v_atom = -0.5 * (
            (diff_v ** 2 / sigma_sq_lig).sum(dim=1) + diff_v.shape[1] * log_2pi_sig
        )

        return (scatter_mean(log_p_pos_atom, lig_mask, dim=0),
                scatter_mean(log_p_v_atom,   lig_mask, dim=0))

    @property
    def total_timesteps(self) -> int:
        return self._inner.T

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – data pre-processing
    # ------------------------------------------------------------------

    def get_ligand_and_pocket(self, data):
        return self._ddpm_module.get_ligand_and_pocket(data)

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – required attributes
    # ------------------------------------------------------------------

    @property
    def atom_nf(self) -> int:
        return self._inner.atom_nf

    @property
    def n_dims(self) -> int:
        return self._inner.n_dims

    # ------------------------------------------------------------------
    # Virtual-node support
    # ------------------------------------------------------------------

    @property
    def virtual_nodes(self) -> bool:
        return getattr(self._ddpm_module, 'virtual_nodes', False)

    @property
    def virtual_atom(self):
        return getattr(self._ddpm_module, 'virtual_atom', None)

    # ------------------------------------------------------------------
    # nn.Module overrides – route to the trainable inner model only
    # so that optimizer and gradient-clipping see only ConditionalDDPM.
    # ------------------------------------------------------------------

    def parameters(self, recurse=True):
        return self._inner.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self._inner.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def train(self, mode=True):
        self._inner.train(mode)
        return self

    def eval(self):
        self._inner.eval()
        return self

    # ------------------------------------------------------------------
    # Pass-through to the Lightning module for validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        return self._ddpm_module.validation_step(batch, batch_idx)

    @property
    def dataset_info(self):
        return self._ddpm_module.dataset_info
