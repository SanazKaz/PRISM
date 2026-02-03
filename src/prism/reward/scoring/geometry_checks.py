"""PoseBustersGeometryReward

Adapted from PoseBusters (Buttenschoen et al.) for Reinforcement Learning.
Uses the original PoseBusters check_geometry function directly.

Checks:
    1. Bond lengths (1-2 interactions) - within expected bounds
    2. Bond angles (1-3 interactions) - within expected bounds  
    3. Steric clashes (non-bonded) - no VDW overlap

Score = exp(-penalty) where penalty is based on violation counts.
This provides smooth gradients - small improvements always help.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from posebusters.modules.distance_geometry import check_geometry
from src.prism.reward.scorer import BaseReward


class PoseBustersGeometryReward(BaseReward):
    """
    Reward based on PoseBusters geometry checks.
    
    Uses exponential penalty approach for smooth gradients.
    Each violation contributes to the penalty, so small improvements
    always result in score improvements.
    """
    
    def __init__(
        self, 
        threshold_bad_bond_length: float = 0.2,
        threshold_bad_angle: float = 0.2,
        threshold_clash: float = 0.2,
        penalty_scale: float = 2.0 # previously 2.0
    ):
        """
        Args:
            threshold_bad_bond_length: Relative tolerance for bonds (0.2 = 20%)
            threshold_bad_angle: Relative tolerance for angles (0.2 = 20%)
            threshold_clash: Relative tolerance for clashes (0.2 = 20%)
            penalty_scale: Controls harshness of penalty (higher = harsher)
        """
        super().__init__()
        self.threshold_bond = threshold_bad_bond_length
        self.threshold_angle = threshold_bad_angle
        self.threshold_clash = threshold_clash
        self.scale = penalty_scale

    @property
    def name(self) -> str:
        return "geometry_checks"
    
    @property
    def increase_weight_after_epoch(self) -> Optional[int]:
        return None

    @property
    def increased_weight_multiplier(self) -> float:    
        return None

    def _safe_value(self, value) -> float:
        """
        Safely extract value, handling NaN.
        
        Returns 0 if NaN.
        """
        # NaN check (NaN != NaN)
        if value != value:
            return 0
        return float(value)

    def score_mol(self, mol: Mol) -> float:
        """
        Score a single molecule based on geometry checks.
        
        Uses exponential penalty approach for smooth gradients.
        Small improvements in geometry always improve the score.
        
        Returns:
            exp(-penalty) where penalty scales with number of violations.
            0.0 if molecule is invalid.
        """
        if mol is None:
            return 0.0
        
        # Call Martin's check_geometry and get the results
        results = check_geometry(
            mol,
            threshold_bad_bond_length=self.threshold_bond,
            threshold_bad_angle=self.threshold_angle,
            threshold_clash=self.threshold_clash
        )
        
        r = results["results"]
        # dict with keys: df_bonds, df_angles, df_clashes \
        # and values are pandas DataFrames with columns:
        details = results.get("details", {})
        df_bonds = details.get("bonds", pd.DataFrame())
        df_angles = details.get("angles", pd.DataFrame())
        df_clashes = details.get("clash", pd.DataFrame())
        
        total_penalty = 0.0
        
        # Sum bond deviations (percent_error is signed: negative = too short, positive = too long)
        for _, row in df_bonds.iterrows():
            bond_pen = abs(row["percent_error"])
            if bond_pen > self.threshold_bond:
                total_penalty += bond_pen * 1.0 # same as before - harsher worked better than allowing for the threshold
                
        # Sum angle deviations (bound_absolute_percent_error is already absolute)
        for _, row in df_angles.iterrows():
            ba_pen = row["bound_absolute_percent_error"]
            if ba_pen > self.threshold_angle:
                total_penalty += ba_pen * 0.5 # same as before - harsher worked better than allowing for the threshold
       
        # Sum clash deviations (bound_percent_error is negative for violations)
        for _, row in df_clashes.iterrows():
            clash_pen = row["bound_percent_error"]
            if clash_pen < -self.threshold_clash: # negative value means too close
                total_penalty += abs(clash_pen) * 1.5 # same as before - harsher worked better than allowing for the threshold
                
        
        score = np.exp(-total_penalty * self.scale)
        
        
        return float(score)

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculate geometry scores for a batch of molecules.
        
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
                
        print(f"Geometry checks scores: {scores}")
        return torch.tensor(scores, dtype=torch.float32)