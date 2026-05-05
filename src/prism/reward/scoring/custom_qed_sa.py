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
    Custom QED reward with explicit fused ring and aliphatic ring penalties,
    an aromatic ring bonus, and normalisation to [0, 1].

    Uses standard QED weights (no ROTB modification - original weights are
    pharmacologically grounded). Fused ring and aliphatic ring penalties are
    applied multiplicatively on top of the capped QED score. An aromatic
    bonus is applied last, encouraging inclusion of at least one aromatic
    ring consistent with reference ligand profiles (median AroR = 1.31).

    Sequence:
        1. Compute raw QED
        2. Cap at QED_CAP (prevents chasing QED=1.0, a mathematical artifact)
        3. Apply fused ring penalty  : exp(-0.5 * max(0, n_fused - 3))
        4. Apply aliphatic ring penalty: exp(-0.5 * max(0, n_ali - 2))
        5. Apply aromatic bonus      : exp(+0.5) if >= 1 aromatic ring, else 1.0
        6. Clip to QED_CAP (bounds the output after bonus)
        7. Normalise by QED_CAP -> maps [0, QED_CAP] to [0, 1]

    Fused ring penalty  : free up to 3, exp(-0.5 * (n - 3)) above that
    Aliphatic ring penalty: free up to 2, exp(-0.5 * (n - 2)) above that
    Aromatic bonus      : exp(+0.5) ~ 1.65x multiplier for >= 1 aromatic ring
    Output cap          : QED_CAP = 0.70 before normalisation

    Range: [0, 1]
    """

    QED_CAP = 0.70
    FUSED_RING_FREE = 3   # rings allowed before penalty kicks in
    ALI_RING_FREE   = 2   # rings allowed before penalty kicks in

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

    def _aromatic_bonus_multiplier(self, mol: Chem.Mol) -> float:
        """
        Smooth exponential bonus encouraging at least one aromatic ring.

        Returns exp(+0.5) ~ 1.65 if >= 1 aromatic ring present,
        1.0 otherwise (no penalty for absence, only reward for presence).

        Applied last, after penalties, so the bonus acts on the
        already-corrected score. Final clip to QED_CAP bounds the output.
        """
        n_aro = Lipinski.NumAromaticRings(mol)
        if n_aro >= 1:
            return float(np.exp(_K))
        return 1.0

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []

        for mol in molecules:
            try:
                # 1. Standard QED with original weights
                qed_score = QED.qed(mol)

                # 2. Cap: no reward for chasing QED perfection
                qed_score = min(qed_score, self.QED_CAP)

                # 3. Multiplicative penalties for undesirable ring complexity
                qed_score *= self._fused_ring_multiplier(mol)
                qed_score *= self._aliphatic_ring_multiplier(mol)

                # 4. Aromatic bonus applied last: rewards inclusion of
                #    aromatic character without penalising its absence
                qed_score *= self._aromatic_bonus_multiplier(mol) # NEW

                # 5. Clip after bonus so output stays bounded
                qed_score = min(qed_score, self.QED_CAP)

                # 6. Normalise to [0, 1] for consistent reward scaling
                qed_score /= self.QED_CAP

                scores.append(float(qed_score))

            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)


class CustomSAScoreReward(BaseReward):
    """
    Synthetic Accessibility reward with an upper cap to prevent the model
    from chasing trivially small, perfectly synthesisable molecules.

    Raw SA score range: 1 (easy) to 10 (hard).
    Normalisation     : (10 - sa) / (10 - SA_RAW_CAP) maps [SA_RAW_CAP, 10]
                        to [1, 0].
    Cap               : applied at SA_RAW_CAP (default 4.0, matching the
                        mean SA of reference ligands in the CrossDocked test
                        set). Molecules with SA <= SA_RAW_CAP all receive
                        the maximum reward of 1.0, removing gradient pressure
                        to keep shrinking or simplifying scaffolds beyond
                        what is pharmacologically meaningful.

    Range: [0, 1]
    """

    SA_RAW_CAP = 4.0  # raw SA scores at or below this are treated as perfect

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

                # Normalise: SA=SA_RAW_CAP -> 1.0, SA=10 -> 0.0
                normalised = (10.0 - sa_raw) / (10.0 - self.SA_RAW_CAP)
                normalised = float(np.clip(normalised, 0.0, 1.0))

                scores.append(normalised)

            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)