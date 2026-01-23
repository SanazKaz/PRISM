"""PoseBustersFlatnessReward

Adapted from PoseBusters (Buttenschoen et al.) for Reinforcement Learning.
Uses the original PoseBusters check_flatness function directly.

Checks:
    1. Flat systems that are bent (Aromatic rings, Double bonds) -> should be flat
    2. Non-flat systems that are flat (Aliphatic 6-rings) -> should be puckered

Score = exp(-penalty) where penalty is based on deviation amounts.
This provides smooth gradients - small improvements always help.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import rdMolDescriptors

from posebusters.modules.flatness import check_flatness, flat, nonflat
from src.prism.reward.scorer import BaseReward


class PoseBustersFlatnessReward(BaseReward):
    """
    Reward based on PoseBusters flatness checks.
    
    Uses exponential penalty approach for smooth gradients.
    Penalty is proportional to how much the deviation exceeds/falls below threshold.
    """
    
    def __init__(self, threshold_flatness: float = 0.1, penalty_scale: float = 5.0):
        """
        Args:
            threshold_flatness: Distance (Angstrom) cutoff for planarity.
                                Flat systems must have deviation < this.
                                Non-flat systems must have deviation >= this.
            penalty_scale: Multiplier for the penalty (higher = harsher)
        """
        super().__init__()
        self.threshold = threshold_flatness
        self.scale = penalty_scale

    @property
    def name(self) -> str:
        return "flatness_checks"
    
    @property
    def increase_weight_after_epoch(self) -> Optional[int]:
        return None

    @property
    def increased_weight_multiplier(self) -> float:
        return None

    def _has_rings(self, mol: Mol) -> bool:
        """Check if molecule contains any ring systems."""
        return rdMolDescriptors.CalcNumRings(mol) > 0

    def score_mol(self, mol: Mol) -> float:
        """
        Score a single molecule based on flatness checks.
        
        Uses exponential penalty based on actual deviation amounts.
        Small improvements in geometry always improve the score.
        
        Returns:
            0.0 if no rings present
            exp(-penalty) otherwise, where penalty scales with deviation
        """
        if mol is None:
            return 0.0
        
        # Check if molecule has any rings at all
        if not self._has_rings(mol):
            return 0.0
        
        total_penalty = 0.0
        flat_max_distances = []
        nonflat_max_distances = []
        
        # Check flat systems (aromatics, double bonds) - should be flat
        flat_results = check_flatness(
            mol, 
            threshold_flatness=self.threshold,
            flat_systems=flat,
            check_nonflat=False
        )
        
        # Extract deviation details for flat systems
        if "details" in flat_results:
            flat_max_distances = flat_results["details"].get("max_distance", [])
            for deviation in flat_max_distances:
                # Penalize if deviation > threshold (too bent)
                if deviation > self.threshold:
                    total_penalty += (deviation - self.threshold) * self.scale
        
        # Check non-flat systems (aliphatic 6-rings) - should be puckered
        nonflat_results = check_flatness(
            mol,
            threshold_flatness=self.threshold,
            flat_systems=nonflat,
            check_nonflat=True
        )
        
        # Extract deviation details for non-flat systems
        if "details" in nonflat_results:
            nonflat_max_distances = nonflat_results["details"].get("max_distance", [])
            for deviation in nonflat_max_distances:
                # Penalize if deviation < threshold (too flat)
                if deviation < self.threshold:
                    total_penalty += (self.threshold - deviation) * self.scale
        
        # If no systems were checked, return 0.5 (neutral)
        flat_checked = flat_results.get("results", {}).get("num_systems_checked", 0)
        nonflat_checked = nonflat_results.get("results", {}).get("num_systems_checked", 0)
        
        # Handle NaN
        if flat_checked != flat_checked:
            flat_checked = 0
        if nonflat_checked != nonflat_checked:
            nonflat_checked = 0
            
        if flat_checked + nonflat_checked == 0:
            return 0.5
        
        print(f"Flat systems found: {len(flat_max_distances)}, deviations: {flat_max_distances}")
        print(f"Non-flat systems found: {len(nonflat_max_distances)}, deviations: {nonflat_max_distances}")
        print(f"Total penalty before exp: {total_penalty}")
        
        # Exponential decay: 0 penalty -> 1.0, more penalty -> lower score
        score = np.exp(-total_penalty)
        
        return float(score)

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculate flatness scores for a batch of molecules.
        
        Args:
            molecules: List of RDKit Mol objects
            dataset_info: Optional dataset metadata (unused)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Tensor of scores, shape (len(molecules),)
        """
        scores = []
        for mol in molecules:
            try:
                score = self.score_mol(mol)
                scores.append(score)
            except Exception:
                scores.append(0.0)
                
        print(f"Flatness checks scores: {scores}")
        return torch.tensor(scores, dtype=torch.float32)