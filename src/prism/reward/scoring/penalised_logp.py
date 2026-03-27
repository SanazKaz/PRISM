import torch
import numpy as np
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from src.models.diffsbdd.analysis.SA_Score.sascorer import calculateScore

from src.prism.reward.scorer import BaseReward

RDLogger.DisableLog('rdApp.*')

_MIN_SCORE = -10.0
_MAX_SCORE =  10.0


class PenalisedLogP(BaseReward):
    """
    Penalised logP reward for single-objective RL benchmarking.

    Defined as:
        penalised_logP = logP(m) - SA(m) - cycle(m)

    where cycle(m) = max(largest_ring_size - 6, 0).

    The raw score is linearly normalised to [0, 1] using fixed bounds
    [_MIN_SCORE, _MAX_SCORE] and clipped, so extreme molecules saturate
    cleanly at 0.0 or 1.0. Invalid molecules return 0.0.

    Adapted from MolDQN (You et al., 2018) and the Junction Tree VAE
    (Jin et al., 2018). Higher values are better.

    References
    ----------
    You et al. (2018) Graph Convolutional Policy Network...
    Jin et al. (2018) Junction Tree Variational Autoencoder...
    """

    @property
    def name(self) -> str:
        return "penalised_logp"

    @staticmethod
    def _get_largest_ring_size(mol: Chem.Mol) -> int:
        """Returns the largest ring size in the molecule, or 0 if acyclic."""
        cycle_list = mol.GetRingInfo().AtomRings()
        return max((len(ring) for ring in cycle_list), default=0)

    @staticmethod
    def _penalised_logp_raw(mol: Chem.Mol) -> float:
        """Computes raw penalised logP for a single valid molecule."""
        log_p = Descriptors.MolLogP(mol)
        sa_score = calculateScore(mol)
        cycle_score = max(PenalisedLogP._get_largest_ring_size(mol) - 6, 0)
        return log_p - sa_score - cycle_score

    @staticmethod
    def _normalise(raw: float) -> float:
        """Linearly normalises raw score to [0, 1] and clips."""
        normalised = (raw - _MIN_SCORE) / (_MAX_SCORE - _MIN_SCORE)
        return float(np.clip(normalised, 0.0, 1.0))

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Compute normalised penalised logP for a batch of molecules.

        Parameters
        ----------
        molecules : List[Chem.Mol]
            Batch of RDKit molecule objects. None entries score 0.0.

        Returns
        -------
        torch.Tensor
            1D tensor of penalised logP values in [0, 1], shape (N,).
            Invalid molecules return 0.0.
        """
        scores = []
        for mol in molecules:
            if mol is None:
                scores.append(0.0)
                continue
            try:
                raw = self._penalised_logp_raw(mol)
                scores.append(self._normalise(raw))
            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)