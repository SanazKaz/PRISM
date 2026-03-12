import torch
from typing import List


from src.models.diffsbdd.analysis.SA_Score.sascorer import calculateScore
from src.prism.reward.scorer import BaseReward

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, GraphDescriptors

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

class QEDReward(BaseReward):
    """
    Calculates Quantitative Estimation of Drug-likeness (QED).
    Range: [0, 1] (Higher is better)
    Uses custom weights to reduce penalty for ring fusion.
    """
    @property
    def name(self) -> str:
        return "qed"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        # Custom weights: (MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS)
        custom_weights = (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)
        scores = []
        
        for mol in molecules:
            try:
                score = QED.qed(mol, w=custom_weights)
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        
        return torch.tensor(scores)


class SAScoreReward(BaseReward):
    """
    Calculates Synthetic Accessibility Score.
    Original SA is 1 (easy) to 10 (hard).
    We normalize it to [0, 1] where 1 is easiest.
    """
    @property
    def name(self) -> str:
        return "sa_score"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                sa_raw = calculateScore(mol)
                # Transform: 1 -> 1.0, 10 -> 0.0
                normalized = (10 - sa_raw) / 9
                # Clamp output to ensure [0, 1] range
                normalized = max(0.0, min(1.0, normalized))
                scores.append(normalized)
            except Exception:
                scores.append(0.0)
                
        return torch.tensor(scores)


class LipinskiReward(BaseReward):
    """
    Calculates how many Lipinski rules are satisfied.
    Range: [0, 5] (Integer steps)
    """
    @property
    def name(self) -> str:
        return "lipinski"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                rule_1 = Descriptors.ExactMolWt(mol) < 500
                rule_2 = Lipinski.NumHDonors(mol) <= 5
                rule_3 = Lipinski.NumHAcceptors(mol) <= 10
                logp = Crippen.MolLogP(mol)
                rule_4 = (-2 <= logp <= 5)
                rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
                
                score = sum([rule_1, rule_2, rule_3, rule_4, rule_5])
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        return torch.tensor(scores)


class BertzReward(BaseReward):
    """
    Calculates Bertz Complexity.
    Range: Unbounded positive float.
    """
    @property
    def name(self) -> str:
        return "bertz"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                scores.append(GraphDescriptors.BertzCT(mol))
            except Exception:
                scores.append(0.0)
        return torch.tensor(scores)