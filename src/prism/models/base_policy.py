from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseDiffusionPolicy(nn.Module, ABC):
    """
    Abstract interface that every diffusion model must implement to plug into
    the PRISM PPO training loop.

    The PPO infrastructure calls exactly three things on the policy:
      1. sample_given_pocket  – used by RolloutCollector during experience collection
      2. log_p_zs_given_zt   – used by the PPO loss during the update step
      3. get_ligand_and_pocket – used by PPOAlgorithm.train_step to pre-process
                                 the raw pocket batch from the dataloader

    Subclasses must also expose atom_nf and n_dims so that the collector can
    build correctly-shaped empty tensors when no valid samples are produced.
    """

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_given_pocket(self, pocket, num_nodes_lig, **kwargs):
        """
        Run the full reverse diffusion trajectory conditioned on a pocket.

        Args:
            pocket (dict): Pocket data batch with keys 'x', 'one_hot', 'size', 'mask'.
            num_nodes_lig (Tensor): [n_samples] number of heavy atoms per ligand.

        Returns:
            xh_lig       (Tensor): [N_atoms, n_dims + atom_nf] final ligand state.
            xh_pocket    (Tensor): [N_pocket_atoms, n_dims + pocket_nf] pocket state.
            lig_mask     (Tensor): [N_atoms] molecule index per ligand atom.
            pocket_mask  (Tensor): [N_pocket_atoms] molecule index per pocket atom.
            mol_log_probs (list[Tensor]): per-step log π(z_s | z_t), one per timestep.
            z_states     (list[Tensor]): latent z_t snapshots, one per timestep.
        """

    # ------------------------------------------------------------------
    # Log-probability (called during PPO update)
    # ------------------------------------------------------------------

    @abstractmethod
    def log_p_zs_given_zt(self, s, t, z_t, z_s, xh_pock, lig_mask, poc_mask):
        """
        Compute log π_θ(z_s | z_t, pocket) using the *current* model weights.

        This is the on-policy re-evaluation needed for the PPO importance ratio.

        Args:
            s        (Tensor): [n_mols, 1] normalised source timestep (s = t-1).
            t        (Tensor): [n_mols, 1] normalised target timestep.
            z_t      (Tensor): [N_atoms, n_dims + atom_nf] latent at step t.
            z_s      (Tensor): [N_atoms, n_dims + atom_nf] latent at step s (from buffer).
            xh_pock  (Tensor): [N_pocket_atoms, ...] pocket features.
            lig_mask (Tensor): [N_atoms] molecule index, re-indexed to 0…n_mols-1.
            poc_mask (Tensor): [N_pocket_atoms] molecule index, re-indexed.

        Returns:
            Tensor: [n_mols] per-molecule log-probability.
        """

    # ------------------------------------------------------------------
    # Data pre-processing
    # ------------------------------------------------------------------

    @abstractmethod
    def get_ligand_and_pocket(self, data):
        """
        Convert a raw dataloader batch into (ligand_dict, pocket_dict, names).
        """

    # ------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def atom_nf(self) -> int:
        """Number of atom-type features in the ligand representation."""

    @property
    @abstractmethod
    def n_dims(self) -> int:
        """Number of spatial dimensions (typically 3)."""

    # ------------------------------------------------------------------
    # Optional virtual-node support (used by build_molecules_from_batch)
    # ------------------------------------------------------------------

    @property
    def virtual_nodes(self) -> bool:
        return False

    @property
    def virtual_atom(self):
        return None
