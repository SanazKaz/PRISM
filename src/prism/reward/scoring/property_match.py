import json
import numpy as np
import torch
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from src.prism.reward.scorer import BaseReward


class Property2DReward(BaseReward):
    """
    Rewards molecules that match reference 2D property distributions.
    
    Uses data-driven ranges (mean ± multiplier * std) from reference dataset.
    Applies harsh penalty for rings larger than 6 atoms.
    """
    
    # How many std devs to allow (lower = stricter)
    RANGE_MULTIPLIERS = {
        'MW': 1.5,          # Moderate - config already controls node count
        'AroR_C': 1.5,      # STRICT - model underproduces, need precise match
        'AliR_C': 0.8,      # STRICT - model overproduces, need precise match
        'SA': 2.0,          # FAIRLY STRICT - important for synthesizability
        'HetA_C': 1.2,      # FAIRLY STRICT - controls composition
        'RotB_C': 1.0,      # STRICT - model makes too many, need to reduce
        'NHOH_C': 1.2,      # FAIRLY STRICT - important for binding
        'ChiA_C': 1.0,      # STRICT - model makes TOO MANY, must reduce
    }
    
    # Property importance weights
    WEIGHTS = {
        'MW': 0.8,          # Low - config handles this
        'AroR_C': 4,      # CRITICAL - model doesn't make enough
        'AliR_C': 4.0,      # CRITICAL - model makes too many
        'SA': 2.0,          # VERY IMPORTANT - fight complexity
        'HetA_C': 2.0,      # IMPORTANT - controls composition
        'RotB_C': 3.0,      # CRITICAL - must reduce drastically
        'NHOH_C': 3.0,      # VERY IMPORTANT - binding interactions
        'ChiA_C': 3.0,      # VERY IMPORTANT - reduce chiral centers
    }
    
    # Hard constraint: no rings larger than this
    MAX_RING_SIZE = 6
    LARGE_RING_PENALTY = 0.5  # Subtract this from final score
    MIN_FSP3 = 0.4
    MAX_FSP3 = 0.5
    FSP3_PENALTY = 0.5
    
    def __init__(self, reference_json_path: str, target_name: str):
        """
        Args:
            reference_json_path: Path to JSON with all target 2D property stats
            target_name: Which target to load (e.g., 'EGFR', 'BRD4_BD1')
        """
        with open(reference_json_path, 'r') as f:
            all_stats = json.load(f)
        
        if target_name not in all_stats:
            raise ValueError(f"Target {target_name} not found in {reference_json_path}")
        
        target_data = all_stats[target_name]
        
        # Extract ranges for each property
        self.ranges = {}
        for prop_name in self.RANGE_MULTIPLIERS.keys():
            if prop_name in target_data:
                prop_data = target_data[prop_name]
                mean = prop_data['mean']
                std = prop_data['std']
                multiplier = self.RANGE_MULTIPLIERS[prop_name]
                
                # Define acceptable range as mean ± (multiplier × std)
                self.ranges[prop_name] = {
                    'mean': mean,
                    'std': std,
                    'lower': mean - multiplier * std,
                    'upper': mean + multiplier * std,
                    'target': prop_data.get('median', mean),  # Prefer median as target
                    'multiplier': multiplier
                }
        
        self.target_name = target_name
        print(f"[Property2D] Loaded stats for {target_name}: {len(self.ranges)} properties")
        
        # Print ranges for debugging
        for prop, range_info in self.ranges.items():
            print(f"  {prop}: [{range_info['lower']:.2f}, {range_info['upper']:.2f}] "
                  f"(target={range_info['target']:.2f})")
    
    @property
    def name(self) -> str:
        return "property_2d"
    
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
            has_large_ring = largest_ring > self.MAX_RING_SIZE
            
            return has_large_ring, largest_ring
            
        except Exception as e:
            print(f"[Property2D] Error checking ring sizes: {e}")
            return False, 0
    
    def _check_fsp3(self, mol: Chem.Mol) -> float:
        """
        Check if molecule has Fsp3 within acceptable range [0.4, 0.5].
        Returns penalty amount (0.0 if in range, negative if outside).
        
        Returns:
            Penalty to subtract from final score (0.0 to -0.5)
            
        range selected from literature
        - https://www.sciencedirect.com/science/article/pii/S135964462030297X
        """
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    
        # If within [0.4, 0.5], no penalty (multiplier 1.0)
        if self.MIN_FSP3 <= fsp3 <= self.MAX_FSP3:
            return 1.0
        
        # Calculate how far outside the range it is
        dist = min(abs(fsp3 - self.MIN_FSP3), abs(fsp3 - self.MAX_FSP3))
        
        # Exponential decay: score drops by half for every 0.1 deviation
        # Adjust the 0.05 to make it steeper or shallower
        multiplier = np.exp(-dist / 0.05)
        
        return float(multiplier)
    
    
    def _get_ring_penalty(self, mol: Chem.Mol) -> float:
        """
        Returns a MULTIPLIER (0.0 to 1.0).
        1.0 = valid rings, <1.0 = contains large rings.
        """
        has_large_ring, largest_ring_size = self._check_ring_sizes(mol)
        
        if not has_large_ring:
            return 1.0
        
        # Penalty scales with how much the limit is exceeded
        # e.g., if limit is 6: 
        # 7 atoms -> exp(-0.5) = 0.60
        # 8 atoms -> exp(-1.0) = 0.36
        excess = largest_ring_size - self.MAX_RING_SIZE
        multiplier = np.exp(-0.5 * excess)
        
        return float(multiplier)
    
    def _count_chiral_centers(self, mol: Chem.Mol) -> int:
        """Count number of chiral centers in molecule."""
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            return len(chiral_centers)
        except Exception as e:
            return 0
    
    def _calculate_properties(self, mol: Chem.Mol) -> dict:
        """Calculate all 2D properties for a molecule."""
        try:
            from rdkit.Chem import RDConfig
            import os
            import sys
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            
            props = {
                'MW': Descriptors.MolWt(mol),
                'AroR_C': Lipinski.NumAromaticRings(mol),
                'AliR_C': Lipinski.NumAliphaticRings(mol),
                'SA': sascorer.calculateScore(mol),
                'HetA_C': Lipinski.NumHeteroatoms(mol),
                'RotB_C': Lipinski.NumRotatableBonds(mol),
                'NHOH_C': Lipinski.NumHDonors(mol),
                'ChiA_C': self._count_chiral_centers(mol),
                'FSP3': rdMolDescriptors.CalcFractionCSP3(mol)
            }
            return props
            
        except Exception as e:
            print(f"[Property2D] Error calculating properties: {e}")
            return None
    
    def _property_score(self, mol_value: float, prop_name: str) -> float:
        """
        Calculate score for a single property based on acceptable range.
        
        Args:
            mol_value: Property value for the molecule
            prop_name: Name of the property
            
        Returns:
            Score contribution (already weighted)
        """
        range_info = self.ranges[prop_name]
        weight = self.WEIGHTS[prop_name]
        
        mean = range_info['mean']
        lower = range_info['lower']
        upper = range_info['upper']
        target = range_info['target']
        std = range_info['std']
        
        # Case 1: Within acceptable range [lower, upper]
        if lower <= mol_value <= upper:
            # Calculate distance from ideal target
            distance_from_target = abs(mol_value - target)
            
            # Maximum possible distance while still in range
            max_distance = max(abs(upper - target), abs(lower - target))
            
            # Score between 0.8 and 1.0 based on distance from target
            # At target: score = 1.0
            # At boundary: score = 0.8
            if max_distance > 0:
                normalized_distance = distance_from_target / max_distance
                score = 1.0 - 0.2 * normalized_distance
            else:
                score = 1.0
            
            return score * weight
        
        # Case 2: Below acceptable range
        elif mol_value < lower:
            excess = lower - mol_value
            penalty = excess / std
            score = max(0.0, 1.0 - penalty * 0.5)
            return score * weight
        
        # Case 3: Above acceptable range
        else:  # mol_value > upper
            excess = mol_value - upper
            penalty = excess / std
            score = max(0.0, 1.0 - penalty * 0.5)
            return score * weight
    
    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []
        
        for mol in molecules:
            if mol is None:
                scores.append(0.0); continue
                
            mol_props = self._calculate_properties(mol)
            if mol_props is None:
                scores.append(0.0); continue
                
            # 1. Base Property Score (0.0 to 1.0)
            total_score = 0.0
            total_weight = 0.0
            for prop_name in self.RANGE_MULTIPLIERS.keys():
                if prop_name in self.ranges:
                    total_score += self._property_score(mol_props[prop_name], prop_name)
                    total_weight += self.WEIGHTS[prop_name]
            
            base_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # 2. Ring Size Penalty (Additive)
            # Instead of multiplying, we subtract a small value.
            # 1.0 = valid, 0.6 = one-atom excess. Let's convert that to a -0.4 penalty.
            ring_mult = self._get_ring_penalty(mol)
            ring_penalty = (1.0 - ring_mult) * 0.5  # Max penalty of -0.5
            
            # Final Score: Balanced and Clamped
            # This prevents one bad ring from making a "great" molecule look like "zero."
            final_score = max(0.0, base_score - ring_penalty)
            
            scores.append(float(final_score))
            
        return torch.tensor(scores, dtype=torch.float32)