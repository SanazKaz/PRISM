from typing import List, Optional

import json
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import Crippen

from src.prism.reward.scorer import BaseReward

class Property2DReward(BaseReward):
    """
    Rewards molecules that match reference 2D property distributions.

    Uses Gaussian scoring centred on the reference mean for each property,
    with sigma = 0.5 * reference std. This gives maximum score at the mean
    and smooth decay outward, providing gradient signal throughout training.

    Also includes a stepped aromatic ring bonus targeting the per-target
    reference mean AroR_C, combined into the final score so weighting is
    handled in one place.

    MW is commented out - node count is controlled by the diffusion config.
    Fsp3 is calculated but intentionally excluded from scoring for now.


    adapted from 
    https://github.com/jianingli-purdue/Benchmarking_gene_model/blob/main/Picture_drawing.ipynb
    Yang et al https://pubs.acs.org/doi/10.1021/acs.jmedchem.5c01706

    """

    SIGMA_MULTIPLIER = 0.7

    WEIGHTS = {
        # 'MW': 0.8,        # Commented out - controlled by node count config
        'AroR_C': 4.0,      # CRITICAL - model underproduces aromatic rings
        'AliR_C': 4.5,      # CRITICAL - model overproduces aliphatic rings
        'SA':     2.5,       # VERY IMPORTANT - synthesizability
        'HetA_C': 2.0,       # IMPORTANT - heteroatom composition
        'RotB_C': 3.5,       # CRITICAL - model overproduces rotatable bonds
        'ChiA_C': 3.0,       # VERY IMPORTANT - model makes too many chiral centres
        'HBD_C': 4.2,       # VERY IMPORTANT - model makes too many HBD atoms
        'HBA_C': 4.0,       # VERY IMPORTANT - model makes too many HBA atoms
        'FusedR_C': 3.5,    # VERY IMPORTANT - model makes too many fused rings
        'LogP_C': 1.0,     # to prevent greasy molecules
    }

    # Aromatic bonus weight - same scale as property weights above
    AROMATIC_BONUS_WEIGHT = 4.0

    # Hard constraint: no rings larger than this
    MAX_RING_SIZE = 6

    def __init__(self, reference_json_path: str, target_name: str):
        """
        Args:
            reference_json_path: Path to JSON with all target 2D property stats.
            target_name: Which target to load (e.g., 'EGFR', 'BRD4_BD1').
        """
        super().__init__()

        with open(reference_json_path, 'r') as f:
            all_stats = json.load(f)

        if target_name not in all_stats:
            raise ValueError(f"Target {target_name} not found in {reference_json_path}")

        target_data = all_stats[target_name]

        # Store mean and sigma per property for Gaussian scoring
        self.gaussian_params = {}
        # 2. But ensure specific "Learning Lanes" are wide enough to feel the slope
        for prop_name in self.WEIGHTS.keys():
            mean = target_data[prop_name]['mean']
            std = target_data[prop_name]['std']
            
            # Calculate initial sigma
            sigma = self.SIGMA_MULTIPLIER * std
            
            # APPLY FLOORS some props have very small stds
            if prop_name in ['SA', 'AroR_C', 'AliR_C', 'ChiA_C', 'LogP_C']:
                sigma = max(sigma, 0.3) 
            
            self.gaussian_params[prop_name] = {'mean': mean, 'sigma': sigma}
                

        # Per-target aromatic ring target from reference mean
        aro_mean = target_data['AroR_C']['mean']
        self.aromatic_target = max(1, round(aro_mean))

        self.target_name = target_name
        print(f"[Property2D] Loaded for {target_name}: {len(self.gaussian_params)} properties")
        print(f"[Property2D] Aromatic bonus target: {self.aromatic_target} rings "
              f"(reference mean={aro_mean:.2f})")

        for prop, params in self.gaussian_params.items():
            print(f"  {prop}: mean={params['mean']:.2f}, sigma={params['sigma']:.2f}")

    @property
    def name(self) -> str:
        return "property_2d"

    @property
    def epoch_weight_schedule(self) -> Optional[int]:
        return None

    @property
    def weight_before_epoch(self) -> Optional[float]:
        return None

    @property
    def weight_after_epoch(self) -> Optional[float]:
        return None

    def _check_ring_sizes(self, mol: Chem.Mol) -> tuple:
        """
        Check if molecule has rings larger than allowed size.

        Returns:
            (has_large_ring: bool, largest_ring_size: int)
        """
        try:
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()

            if not rings:
                return False, 0

            largest_ring = max(len(ring) for ring in rings)
            return largest_ring > self.MAX_RING_SIZE, largest_ring

        except Exception as e:
            print(f"[Property2D] Error checking ring sizes: {e}")
            return False, 0

    def _check_fsp3(self, mol: Chem.Mol) -> float:
        """
        Calculates Fsp3 fraction. Currently disconnected from scoring.
        Kept for future use - do not wire into __call__ yet.
        """
        return float(rdMolDescriptors.CalcFractionCSP3(mol))

    def _get_ring_penalty(self, mol: Chem.Mol) -> float:
        """
        Returns a multiplier (0.0 to 1.0) penalising rings larger than MAX_RING_SIZE.

            7-membered ring -> exp(-0.5) = 0.60
            8-membered ring -> exp(-1.0) = 0.36
        """
        has_large_ring, largest_ring_size = self._check_ring_sizes(mol)

        if not has_large_ring:
            return 1.0

        excess = largest_ring_size - self.MAX_RING_SIZE
        return float(np.exp(-0.5 * excess))

    def _count_chiral_centers(self, mol: Chem.Mol) -> int:
        """Count number of chiral centres in molecule."""
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            return len(chiral_centers)
        except Exception:
            return 0

    def _calculate_properties(self, mol: Chem.Mol) -> dict:
        """Calculate all 2D properties for a molecule."""
        try:
            from rdkit.Chem import RDConfig
            import os
            import sys
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer

            ring_info = mol.GetRingInfo()

            props = {
                # 'MW': Descriptors.MolWt(mol),   # Commented out - controlled by config
                'AroR_C': Lipinski.NumAromaticRings(mol) / 3,
                'AliR_C': Lipinski.NumAliphaticRings(mol) / 4,
                'SA':     sascorer.calculateScore(mol) / 6.0, # based on paper listed above
                'HetA_C': sum(1 for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() not in (1, 6)) / 10,
                'RotB_C': Descriptors.NumRotatableBonds(mol) / 8,
                'ChiA_C': self._count_chiral_centers(mol) / 6,
                'FSP3':   rdMolDescriptors.CalcFractionCSP3(mol),  # Calculated, not scored
                # 'BriA_C': rdMolDescriptors.CalcNumBridgeheadAtoms(mol) / 2,
                'HBD_C': rdMolDescriptors.CalcNumHBD(mol),
                'HBA_C': rdMolDescriptors.CalcNumHBA(mol),
                'FusedR_C': sum(1 for i in range(ring_info.NumRings()) if ring_info.IsRingFused(i)),
                'LogP_C': Crippen.MolLogP(mol) / 5,
            }

            return props

        except Exception as e:
            print(f"[Property2D] Error calculating properties: {e}")
            return None

    def _property_score(self, mol_value: float, prop_name: str) -> float:
        """
        Gaussian score for a single property, weighted by property importance.

        Score = exp(-(mol_value - mean)^2 / (2 * sigma^2)) * weight

        At the reference mean:  score = 1.0 * weight
        At mean ± 1 sigma:      score ≈ 0.607 * weight
        At mean ± 2 sigma:      score ≈ 0.135 * weight
        """
        params = self.gaussian_params[prop_name]
        weight = self.WEIGHTS[prop_name]
        gaussian = np.exp(-((mol_value - params['mean']) ** 2) / (2 * params['sigma'] ** 2))
        return float(gaussian * weight)

    def _aromatic_bonus(self, aro_count: int) -> float:
        """
        Stepped bonus toward per-target reference aromatic ring count.

        Scoring relative to self.aromatic_target:
            >= target:      AROMATIC_BONUS_WEIGHT (full bonus)
            target - 1:     AROMATIC_BONUS_WEIGHT * 0.5
            < target - 1:   0.0
        """
        if aro_count >= self.aromatic_target:
            return self.AROMATIC_BONUS_WEIGHT
        elif aro_count == self.aromatic_target - 1:
            return self.AROMATIC_BONUS_WEIGHT * 0.5
        else:
            return 0.0

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []

        for mol in molecules:
            if mol is None:
                scores.append(0.0)
                continue

            mol_props = self._calculate_properties(mol)
            if mol_props is None:
                scores.append(0.0)
                continue

            # 1. Gaussian property scores normalised by total weight
            total_score  = 0.0
            total_weight = 0.0
            for prop_name in self.WEIGHTS.keys():
                if prop_name in self.gaussian_params:
                    total_score  += self._property_score(mol_props[prop_name], prop_name)
                    total_weight += self.WEIGHTS[prop_name]

            # 2. Aromatic bonus added into the same weighted sum
            aro_count_raw = int(round(mol_props['AroR_C'] * 3)) 
            total_score  += self._aromatic_bonus(aro_count_raw)

            total_weight += self.AROMATIC_BONUS_WEIGHT

            base_score = total_score / total_weight if total_weight > 0 else 0.0

            # 3. Ring size penalty
            ring_mult    = self._get_ring_penalty(mol)
            ring_penalty = (1.0 - ring_mult) * 0.5
            final_score  = max(0.0, base_score - ring_penalty)

            # # Temporary debug - remove before full training run
            # debug_lines = []
            # for prop_name in self.WEIGHTS.keys():
            #     if prop_name in self.gaussian_params and prop_name in mol_props:
            #         raw_score = self._property_score(mol_props[prop_name], prop_name)
            #         debug_lines.append(
            #             f"  {prop_name}={mol_props[prop_name]:.2f} "
            #             f"(mean={self.gaussian_params[prop_name]['mean']:.2f}, "
            #             f"sigma={self.gaussian_params[prop_name]['sigma']:.2f}) "
            #             f"-> weighted={raw_score:.3f}"
            #         )
            # print(f"[Property2D Debug] {Chem.MolToSmiles(mol)}")
            # print("\n".join(debug_lines))
            # print(f"  aromatic_bonus={self._aromatic_bonus(aro_count_raw):.2f}")
            # print(f"  base={base_score:.3f}, ring_penalty={ring_penalty:.3f}, final={final_score:.3f}")
            
            # Apply a threshold to the score to prevent it from being too high
            threshold = 0.65
            if final_score > threshold:
                score = 1.0
            else:
                score = final_score / threshold
            
            scores.append(float(score))


        return torch.tensor(scores, dtype=torch.float32)