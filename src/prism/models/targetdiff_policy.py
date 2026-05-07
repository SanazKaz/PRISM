"""
TargetDiff integration stub.

TargetDiff paper: https://arxiv.org/abs/2303.03543
Source repo:      https://github.com/guanjq/targetdiff

Architecture differences vs DiffSBDD that affect this adapter:
  - Atom types are truly discrete (categorical uniform forward process) rather
    than continuous floats.  The per-step log-prob has two terms:
      * Gaussian for 3D coordinates  →  reconstruct posterior mean from eps
      * Categorical KL for atom types →  q_v_posterior() output (already in log-space)
  - The score model is ScorePosNet3D (UniTransformer backbone), not EGNN.
  - Sampling entry-point is model.sample_diffusion(), not sample_given_pocket().

TODO (in implementation order):
  1. Install / vendor TargetDiff source under src/models/targetdiff/.
  2. Implement _build_pocket_batch()     – convert PRISM pocket dict to TargetDiff format.
  3. Implement sample_given_pocket()     – wrap model.sample_diffusion(); collect z_states
                                          and per-step log_probs.
  4. Implement _coord_log_prob()         – reconstruct Gaussian posterior mean from
                                          predicted eps and evaluate N(z_s; mu, sigma²).
  5. Implement _type_log_prob()          – call model.q_v_posterior() and sum log-probs
                                          over atoms (scatter_mean over lig_mask).
  6. Implement log_p_zs_given_zt()       – combine coord + type log-probs into one scalar
                                          per molecule.
  7. Implement get_ligand_and_pocket()   – parse raw dataloader batch into
                                          (ligand_dict, pocket_dict, names).
  8. Wire up atom_nf / n_dims to match TargetDiff's feature dimensions.
  9. Add checkpoint loading + config plumbing in PPOFineTuner.
"""

import torch
import torch.nn as nn
from src.prism.models.base_policy import BaseDiffusionPolicy


class TargetDiffPolicy(BaseDiffusionPolicy):
    """
    Adapter that wraps TargetDiff's ScorePosNet3D to satisfy BaseDiffusionPolicy.

    Args:
        model: Instantiated ScorePosNet3D from src/models/targetdiff/.
        atom_type_trans: The DiscreteTransition object used during training
                         (provides q_v_posterior for the categorical log-prob).
        num_atom_types (int): Number of heavy-atom type classes.
        n_dims (int): Spatial dimensions (3).
        timesteps (int): Total diffusion timesteps T.
    """

    def __init__(self, model, atom_type_trans, num_atom_types: int, n_dims: int = 3, timesteps: int = 1000):
        super().__init__()
        self._model = model
        self._atom_type_trans = atom_type_trans
        self._num_atom_types = num_atom_types
        self._n_dims = n_dims
        self._timesteps = timesteps

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – sampling
    # ------------------------------------------------------------------

    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        """
        TODO: implement.

        Should:
          1. Convert `pocket` from PRISM format to TargetDiff's compose_context format.
          2. Call self._model.sample_diffusion() or replicate its loop step-by-step so
             we can intercept each (z_t, z_s) pair and compute per-step log-probs.
          3. Return the standard 6-tuple:
             (xh_lig, xh_pocket, lig_mask, pocket_mask, mol_log_probs, z_states)
             where mol_log_probs[i] is a [n_mols] tensor and z_states[i] is [N_atoms, D].
        """
        raise NotImplementedError("TargetDiffPolicy.sample_given_pocket not yet implemented")

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – log-probability
    # ------------------------------------------------------------------

    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """
        TODO: implement.

        The combined per-molecule log-prob is:
            log π(z_s | z_t) = log p_coord(x_s | x_t) + log p_type(v_s | v_t)

        coord term:
            Run a forward pass through self._model to get pred_eps (coords only).
            Reconstruct posterior mean:
              mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * pred_eps)
            Evaluate N(x_s; mu, sigma_t^2 * I) in log-space.
            Sum over atoms, then scatter_mean over lig_mask.

        type term:
            log_v_recon = F.log_softmax(self._model(z_t, pocket, t)[type_output], dim=-1)
            log_vt      = index_to_log_onehot(argmax(z_t[:, n_dims:]), num_classes)
            log_q       = self._atom_type_trans.q_v_posterior(log_v_recon, log_vt, t_int, lig_mask)
            scatter_mean(log_q.sum(dim=-1), lig_mask)

        Returns:
            Tensor [n_mols] – per-molecule log-probability.
        """
        raise NotImplementedError("TargetDiffPolicy.log_p_zs_given_zt not yet implemented")

    # ------------------------------------------------------------------
    # BaseDiffusionPolicy – data pre-processing
    # ------------------------------------------------------------------

    def get_ligand_and_pocket(self, data):
        """
        TODO: implement.

        Convert the raw PRISM dataloader batch into the (ligand, pocket, names)
        triple that PPOAlgorithm.train_step expects.  TargetDiff represents pocket
        atoms with (element, amino-acid type, backbone indicator) features, so the
        feature tensor width differs from DiffSBDD's one-hot.
        """
        raise NotImplementedError("TargetDiffPolicy.get_ligand_and_pocket not yet implemented")

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
    # nn.Module routing
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
