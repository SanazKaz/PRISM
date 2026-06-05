"""OxygenFraction Reward

Sanity-check reward: score = sqrt(n_oxygen / n_heavy_atoms).
Encourages the policy to generate molecules with more oxygen atoms.
Simple, dense signal useful for verifying PPO is learning at all.
"""

from __future__ import annotations
from typing import List

import math
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward


class OxygenFraction(BaseReward):

    @property
    def name(self) -> str:
        return "oxygen_fraction"

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms == 0:
            return 0.0
        n_oxygen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        return math.sqrt(n_oxygen / heavy_atoms)

    def __call__(self, molecules: List[Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                scores.append(self.score_mol(mol))
            except Exception:
                scores.append(0.0)
        return torch.tensor(scores, dtype=torch.float32)
