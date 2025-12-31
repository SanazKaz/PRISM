"""AromaticBonus Reward

Simple reward encouraging aromatic ring incorporation.
Designed as a curriculum learning step before introducing
position-based pharmacophore rewards like FeatureDensityReward.

Usage:
    from src.prism.reward.aromatic_reward import AromaticBonus
    
    reward_fn = AromaticBonus(target_count=2)
    scores = reward_fn(molecules)
"""

from __future__ import annotations

from typing import List

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward


class AromaticBonus(BaseReward):
    """
    Reward function encouraging aromatic ring incorporation.
    
    This provides a dense, easy-to-optimise signal that teaches
    the model to generate molecules containing aromatic rings
    before introducing more complex position-based rewards.
    
    Scoring:
        - >= target_count aromatic rings: 1.0
        - target_count - 1 aromatic rings: 0.5
        - fewer aromatic rings: 0.0
    
    Attributes:
        target_count: Ideal number of aromatic rings for maximum score.
    """
    
    def __init__(self, target_count: int = 2):
        """
        Initialise the AromaticBonus reward.
        
        Args:
            target_count: Ideal number of aromatic rings for full score.
                          Default is 2, which is typical for drug-like molecules.
        """
        super().__init__()
        self.target_count = target_count
        
        print(f"AromaticBonus initialised: target_count={self.target_count}")
    
    @property
    def name(self) -> str:
        """Name of the reward for logging purposes."""
        return "aromatic_counter"
    
    def _count_aromatic_rings(self, mol: Mol) -> int:
        """
        Count fully aromatic rings in a molecule.
        
        A ring is counted as aromatic only if ALL atoms in the ring
        are aromatic (e.g., benzene, pyridine, thiophene).
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Number of fully aromatic rings.
        """
        if mol is None:
            return 0
        
        ring_info = mol.GetRingInfo()
        aromatic_count = 0
        
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_count += 1
        
        return aromatic_count
    
    def score_mol(self, mol: Mol) -> float:
        """
        Score a single molecule based on aromatic ring count.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Score between 0.0 and 1.0.
        """
        count = self._count_aromatic_rings(mol)
        
        if count >= self.target_count:
            return 1.0
        elif count == self.target_count - 1:
            return 0.5
        else:
            return 0.0
    
    def __call__(self, molecules: List[Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculate aromatic bonus for a batch of molecules.
        
        Args:
            molecules: List of RDKit molecule objects.
            dataset_info: Optional dataset metadata (unused but kept for interface).
            **kwargs: Additional arguments (unused but kept for interface).
            
        Returns:
            Tensor of scores with shape (len(molecules),).
        """
        scores = []
        
        for mol in molecules:
            try:
                score = self.score_mol(mol)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)