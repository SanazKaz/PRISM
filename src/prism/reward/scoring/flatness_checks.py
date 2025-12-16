"""PoseBustersFlatnessReward
- Adapted from PoseBusters (Buttenschoen et al.) for Reinforcement Learning
- Checks the planarity of specific substructures using EXACT PoseBusters SMARTS.
- Penalizes:
    1. Flat systems that are bent (Aromatic rings, Double bonds).
    2. Non-flat systems that are flat (Aliphatic rings that should pucker).
    
    adapted from 
"""

from __future__ import annotations
from typing import List, Dict

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward

class PoseBustersFlatnessReward(BaseReward):
    def __init__(self, threshold_flatness: float = 0.1, penalty_scale: float = 5.0):
        """
        Args:
            threshold_flatness: Distance (Angstrom) cutoff. 
                                Flat systems must be < this.
                                Non-flat systems must be > this.
            penalty_scale: Multiplier for the penalty gradient.
        """
        super().__init__()
        self.threshold = threshold_flatness
        self.scale = penalty_scale

        # --- 1. Systems that MUST be flat (from PoseBusters source) ---
        self.flat_systems_smarts = {
            "aromatic_5_membered_rings_sp2": "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1",
            "aromatic_6_membered_rings_sp2": "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1",
            "trigonal_planar_double_bonds": "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)"
        }

        # --- 2. Systems that MUST NOT be flat (from PoseBusters source) ---
        # These capture specific non-aromatic 6-rings (like cyclohexane, piperidine)
        # that must adopt a chair/boat conformation.
        self.nonflat_systems_smarts = {
            "non-aromatic_6_membered_rings": "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1",
            "non-aromatic_6_membered_rings_db03_0": "[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db03_1": "[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db02_0": "[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1",
            "non-aromatic_6_membered_rings_db02_1": "[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1",
        }

        # Compile SMARTS into RDKit objects
        self.flat_patterns = {k: Chem.MolFromSmarts(v) for k, v in self.flat_systems_smarts.items()}
        self.nonflat_patterns = {k: Chem.MolFromSmarts(v) for k, v in self.nonflat_systems_smarts.items()}

        # Verification
        for name, pat in {**self.flat_patterns, **self.nonflat_patterns}.items():
            if pat is None:
                print(f"[!] Warning: Invalid SMARTS for {name}")

    @property
    def name(self) -> str:
        return "flatness_checks"

    def _calculate_planarity_deviation(self, coords: np.ndarray) -> float:
        """
        Calculates the maximum distance of any point from the best-fit plane.
        Uses SVD.
        """
        if coords.shape[0] < 3:
            return 0.0
            
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        
        try:
            # SVD to find plane normal (vector corresponding to smallest singular value)
            u, s, vh = np.linalg.svd(centered)
            normal = vh[2, :] 
        except Exception:
            return 0.0 

        # Distances are projections onto the normal vector
        distances = np.abs(np.dot(centered, normal))
        return float(np.max(distances))

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        
        try:
            conf = mol.GetConformer()
        except ValueError:
            return 0.0

        total_penalty = 0.0
        positions = conf.GetPositions()

        # --- A. Check Flat Systems (Penalty if Deviation > Threshold) ---
        # "It's supposed to be flat, but it's bent."
        for pattern in self.flat_patterns.values():
            matches = mol.GetSubstructMatches(pattern)
            for match_indices in matches:
                group_coords = positions[list(match_indices)]
                deviation = self._calculate_planarity_deviation(group_coords)
                
                if deviation > self.threshold:
                    # Penalize the excess deviation
                    total_penalty += (deviation - self.threshold) * self.scale

        # --- B. Check Non-Flat Systems (Penalty if Deviation < Threshold) ---
        # "It's supposed to be puckered (e.g. chair), but it's flat."
        for pattern in self.nonflat_patterns.values():
            matches = mol.GetSubstructMatches(pattern)
            for match_indices in matches:
                group_coords = positions[list(match_indices)]
                deviation = self._calculate_planarity_deviation(group_coords)
                
                # If deviation is too small (too flat), penalize
                if deviation < self.threshold:
                    total_penalty += (self.threshold - deviation) * self.scale

        # Convert to Score (0.0 to 1.0)
        score = np.exp(-total_penalty)
        return float(score)

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                score = self.score_mol(mol)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        print(f"Flatness checks scores: {scores}")
        return torch.tensor(scores, dtype=torch.float32)