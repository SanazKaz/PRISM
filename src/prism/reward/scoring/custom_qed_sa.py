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
    an aromatic ring term, and normalisation to [0, 1].

    Uses standard QED weights (no ROTB modification - original weights are
    pharmacologically grounded). Ring-complexity and aromaticity terms are
    applied multiplicatively on top of the *normalised* capped QED score.

    All three multipliers are <= 1, so the reward stays monotonically
    increasing in QED across the whole range [0, QED_CAP]. This is deliberate:
    an earlier version applied the aromatic term as an exp(+0.5) ~ 1.65x
    *bonus* and then re-clipped to QED_CAP, which saturated the reward at 1.0
    for every aromatic molecule with QED >= QED_CAP / exp(0.5) = 0.455 -
    i.e. a flat plateau over the entire pharmacologically relevant QED range,
    leaving no gradient for PPO to climb.

    Aromaticity is therefore expressed as a penalty for *absence* rather than
    a bonus for presence. The aromatic/non-aromatic ratio is unchanged
    (exp(0.5) ~ 1.65), so the chemical preference is identical - only the
    ceiling moves, restoring the QED hill.

    Sequence:
        1. Compute raw QED
        2. Cap at QED_CAP (prevents chasing QED=1.0, a mathematical artifact)
        3. Normalise by QED_CAP -> maps [0, QED_CAP] to [0, 1]
        4. Apply fused ring penalty    : exp(-0.5 * max(0, n_fused - 2))
        5. Apply aliphatic ring penalty: exp(-0.5 * max(0, n_ali - 1))
        6. Apply aromatic penalty      : exp(-0.5) if 0 aromatic rings, else 1.0

    Fused ring penalty    : free up to FUSED_RING_FREE, exp(-0.5 * excess) above
    Aliphatic ring penalty: free up to ALI_RING_FREE, exp(-0.5 * excess) above
    Aromatic penalty      : exp(-0.5) ~ 0.61x for molecules with no aromatic
                            ring, consistent with reference ligand profiles
                            (median AroR = 1.31)
    QED cap               : QED_CAP = 0.75 before normalisation

    Range: [0, 1]
    """

    QED_CAP = 0.75
    FUSED_RING_FREE = 2   # rings allowed before penalty kicks in 3 > 2
    ALI_RING_FREE   = 1   # rings allowed before penalty kicks in 2 > 1

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

    def _aromatic_multiplier(self, mol: Chem.Mol) -> float:
        """
        Exponential penalty for molecules lacking any aromatic ring.

        Returns 1.0 if >= 1 aromatic ring present, exp(-0.5) ~ 0.61 otherwise.

        Expressed as a penalty rather than a bonus so the multiplier never
        exceeds 1 and the reward is never clipped - see the class docstring
        for why the previous exp(+0.5) bonus flattened the QED gradient.
        """
        n_aro = Lipinski.NumAromaticRings(mol)
        if n_aro >= 1:
            return 1.0
        return float(np.exp(-_K))

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []

        for mol in molecules:
            try:
                # 1. Standard QED with original weights
                qed_score = QED.qed(mol)

                # 2. Cap: no reward for chasing QED perfection
                qed_score = min(qed_score, self.QED_CAP)

                # 3. Normalise to [0, 1] before the multipliers, so the
                #    penalties scale the full-range score and nothing clips
                qed_score /= self.QED_CAP

                # 4. Multiplicative penalties for undesirable ring complexity
                qed_score *= self._fused_ring_multiplier(mol)
                qed_score *= self._aliphatic_ring_multiplier(mol)

                # 5. Aromatic penalty: molecules with no aromatic ring are
                #    discounted; presence is the neutral case
                qed_score *= self._aromatic_multiplier(mol)

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

    SA_RAW_CAP = 3.5  # raw SA scores at or below this are treated as perfect

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