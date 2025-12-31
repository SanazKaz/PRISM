"""PoseBustersReward (Full Suite)
- Checks Bond Lengths (1-2 interactions)
- Checks Bond Angles (1-3 interactions via distance constraints)
- Checks Steric Clashes (non-bonded interactions)
- adapted  logic from PoseBusters (Buttenschoen et al.) for Reinforcement Learning: https://github.com/maabuu/posebusters
"""

from __future__ import annotations
from typing import List, Optional

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward

class PoseBustersGeometryChecks(BaseReward):
    def __init__(self, 
                 bond_tol: float = 0.2, 
                 angle_tol: float = 0.2, 
                 clash_tol: float = 0.2,
                 penalty_scale: float = 2.0):
        """
        Args:
            bond_tol: Relative tolerance for bond lengths (default 0.2 = 20%)
            angle_tol: Relative tolerance for 1-3 distances (angles)
            clash_tol: Relative tolerance for steric clashes
            penalty_scale: Sharpness of the penalty (higher = harsher)
        """
        super().__init__()
        self.bond_tol = bond_tol
        self.angle_tol = angle_tol
        self.clash_tol = clash_tol
        self.scale = penalty_scale
        
        # Exact parameters used in PoseBusters check_geometry source
        self.bounds_params = {
            "set15bounds": True,
            "scaleVDW": True,
            "doTriangleSmoothing": True,
            "useMacrocycle14config": False,
        }

    @property
    def name(self) -> str:
        return "geometry_checks"

    # @property
    # def increase_weight_after_epoch(self) -> Optional[int]:
    #     """Enable weight increase after epoch 10."""
    #     return 10
    
    # @property
    # def increased_weight_multiplier(self) -> float:
    #     """Increase weight from 0.5 to 0.7 (multiplier of 1.4)."""
    #     return 1.4

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
            
        # --- 1. SANITIZATION (Required for Bounds Matrix) ---
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return 0.0 

        # --- 2. SETUP MATRICES ---
        try:
            # Physics Bounds (Lower Triangle = Min, Upper Triangle = Max)
            # This matrix contains the allowed distances for Bonds (1-2) and Angles (1-3)
            # and the VDW limits for non-bonded atoms.
            bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol, **self.bounds_params)
            
            # Topological Distance (1=Bond, 2=Angle, 3+=Non-bonded)
            topo_dist = Chem.GetDistanceMatrix(mol)
            
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
        except Exception:
            return 0.0

        total_penalty = 0.0
        
        # --- 3. CHECK PAIRWISE GEOMETRY ---
        # Iterate over unique pairs (i < j)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                
                # Actual 3D distance between atoms i and j
                dist = rdMolTransforms.GetBondLength(conf, i, j)
                
                # Limits from RDKit Bounds Matrix
                # bounds[j, i] is LOWER bound
                # bounds[i, j] is UPPER bound
                lower_limit = bounds[j, i]
                upper_limit = bounds[i, j]
                
                # Number of bonds between atoms
                hops = int(topo_dist[i, j])

                violation = 0.0

                # === CASE A: BOND LENGTHS (1-2) ===
                if hops == 1:
                    if dist < lower_limit:
                        # Too Short
                        error = (lower_limit - dist) / lower_limit
                        if error > self.bond_tol: violation = error
                    elif dist > upper_limit:
                        # Too Long
                        error = (dist - upper_limit) / upper_limit
                        if error > self.bond_tol: violation = error
                        
                # === CASE B: BOND ANGLES (1-3) ===
                # PoseBusters checks angles by measuring the distance between the two outer atoms.
                elif hops == 2:
                    if dist < lower_limit:
                        # Angle too closed
                        error = (lower_limit - dist) / lower_limit
                        if error > self.angle_tol: violation = error * 0.5 
                    elif dist > upper_limit:
                        # Angle too open
                        error = (dist - upper_limit) / upper_limit
                        if error > self.angle_tol: violation = error * 0.5

                # === CASE C: STERIC CLASHES (Non-bonded) ===
                elif hops >= 3: 
                    # For clashes, we ONLY check the LOWER bound. 
                    # There is no upper limit on how far apart non-bonded atoms can be.
                    if dist < lower_limit:
                        error = (lower_limit - dist) / lower_limit
                        if error > self.clash_tol: violation = error * 1.5

                total_penalty += violation * self.scale

        # Convert to Score [0.0 - 1.0]
        # exp(-penalty) creates a smooth gradient: 0 penalty -> 1.0 score
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
        print(f"Geometry checks scores: {scores}")
        return torch.tensor(scores, dtype=torch.float32)