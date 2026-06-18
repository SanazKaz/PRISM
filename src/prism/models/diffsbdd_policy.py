import torch
import torch.nn as nn
from pathlib import Path

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
        """Returns (log_p_pos, log_p_v) — Gaussian coord and atom-type channels separately."""
        return self._inner.log_p_zs_given_zt_channels(
            s, t, z_t, z_s, xh_pock, lig_mask, poc_mask)

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
