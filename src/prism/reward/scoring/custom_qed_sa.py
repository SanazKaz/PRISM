import torch
import numpy as np
from typing import List

from src.models.diffsbdd.analysis.SA_Score.sascorer import calculateScore
from src.prism.reward.scorer import BaseReward

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Lipinski, QED, rdMolDescriptors

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Exponential decay constant - shared across ring penalties for consistency
_K = 0.5


class CustomQEDReward(BaseReward):
    """
    Custom QED reward with explicit fused ring and aliphatic ring penalties.

    Uses standard QED weights (no ROTB modification - original weights are
    pharmacologically grounded). Fused ring and aliphatic ring penalties are
    applied multiplicatively on top of the capped QED score.

    Fused ring penalty  : free up to 3, exp(-0.5 * (n - 3)) above that
    Aliphatic ring penalty: free up to 2, exp(-0.5 * (n - 2)) above that
    Output cap          : 0.75 (prevents chasing QED=1.0 which is a
                          mathematical artifact, not a better drug)

    Range: [0, 0.70]
    """

    QED_CAP = 0.70
    FUSED_RING_FREE  = 3   # rings allowed before penalty kicks in
    ALI_RING_FREE    = 2   # rings allowed before penalty kicks in

    @property
    def name(self) -> str:
        return "custom_qed"

    def _fused_ring_count(self, mol: Chem.Mol) -> int:
        """Count number of fused rings in molecule."""
        ring_info = mol.GetRingInfo()
        return sum(
            1 for i in range(ring_info.NumRings())
            if ring_info.IsRingFused(i)
        )

    def _fused_ring_multiplier(self, mol: Chem.Mol) -> float:
        """
        Smooth exponential multiplier penalising excess fused rings.

        Returns 1.0 if fused rings <= FUSED_RING_FREE,
        exp(-0.5 * excess) otherwise.
        """
        n_fused = self._fused_ring_count(mol)
        excess = n_fused - self.FUSED_RING_FREE
        if excess <= 0:
            return 1.0
        return float(np.exp(-_K * excess))

    def _aliphatic_ring_multiplier(self, mol: Chem.Mol) -> float:
        """
        Smooth exponential multiplier penalising excess aliphatic rings.

        Returns 1.0 if aliphatic rings <= ALI_RING_FREE,
        exp(-0.5 * excess) otherwise.
        """
        n_ali = Lipinski.NumAliphaticRings(mol)
        excess = n_ali - self.ALI_RING_FREE
        if excess <= 0:
            return 1.0
        return float(np.exp(-_K * excess))

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []

        for mol in molecules:
            try:
                # 1. Standard QED with original weights
                qed_score = QED.qed(mol)

                # 2. Cap: no reward for chasing QED perfection
                qed_score = min(qed_score, self.QED_CAP)

                # 3. Multiplicative ring penalties applied on top
                qed_score *= self._fused_ring_multiplier(mol)
                qed_score *= self._aliphatic_ring_multiplier(mol)

                scores.append(float(qed_score))

            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)


class CustomSAScoreReward(BaseReward):
    """
    Synthetic Accessibility reward with an upper cap to prevent the model
    from chasing trivially small, perfectly synthesisable molecules.

    Raw SA score range: 1 (easy) to 10 (hard).
    Normalisation     : (10 - sa) / 9  maps this to [0, 1].
    Cap               : applied at SA_CAP_NORMALISED, corresponding to
                        raw SA <= SA_RAW_CAP (default 3.0).

    Molecules with SA <= 3.5 all receive the maximum reward of 1.0,
    removing gradient pressure to keep shrinking/simplifying beyond
    what is pharmacologically meaningful.

    Range: [0, 1]
    """

    SA_RAW_CAP = 3.5  # raw SA scores below this are all treated as perfect

    @property
    def name(self) -> str:
        return "custom_sa_score"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []

        for mol in molecules:
            try:
                sa_raw = calculateScore(mol)

                # Cap: anything easier than SA_RAW_CAP gets full reward
                sa_raw = max(sa_raw, self.SA_RAW_CAP)

                # Normalise: SA=3 -> 1.0, SA=10 -> 0.0
                # Using (10 - sa) / (10 - SA_RAW_CAP) keeps the full [0,1]
                # range across the meaningful part of the SA scale
                normalised = (10.0 - sa_raw) / (10.0 - self.SA_RAW_CAP)
                normalised = float(np.clip(normalised, 0.0, 1.0))

                scores.append(normalised)

            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)