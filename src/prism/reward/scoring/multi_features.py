"""multi_feature.py

Multiplicative pharmacophore reward combining aromatic placement
and H-bond donor/acceptor features.

The multiplicative combination prevents the model from sacrificing
one objective for another - both must be optimised together.

Example scores:
    aromatic=0.05, hbond=0.9 → reward=0.045 (terrible)
    aromatic=0.5, hbond=0.5  → reward=0.25  (decent)
    aromatic=0.7, hbond=0.7  → reward=0.49  (good)
"""

import os
import pickle
from typing import List, Dict, Tuple

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward


class MultiFeatureReward(BaseReward):
    """
    Multiplicative reward combining aromatic hotspot placement and
    H-bond donor/acceptor feature density.
    
    Reward = (aromatic_score ^ aromatic_exp) * (hbond_score ^ hbond_exp)
    
    This formulation forces the model to maintain both objectives
    as tanking either one destroys the total reward.
    
    Attributes:
        cluster_centers: Array of hotspot center coordinates.
        cluster_counts: Consensus counts for each hotspot.
        cluster_features: Feature type labels for each hotspot.
        target_profile: Dict of ideal feature counts per type.
    """
    
    # H-bond feature weights (within the hbond component)
    HBOND_WEIGHTS = {
        'Acceptor': 0.45,
        'Donor': 0.45,
        'NegIonizable': 0.10
    }
    
    def __init__(
        self,
        pkl_path: str,
        # Aromatic parameters
        aromatic_sigma: float = 0.8,
        aromatic_cutoff: float = 2.5,
        n_aromatic_targets: int = 3,
        # H-bond parameters
        hbond_sigma: float = 0.8,
        hbond_cutoff: float = 3.2,
        # Combination parameters
        aromatic_exp: float = 0.5,
        hbond_exp: float = 0.5
    ):
        """
        Initialise the multiplicative feature reward.
        
        Args:
            pkl_path: Path to pickled hotspot data from DBSCAN clustering.
            aromatic_sigma: Gaussian width for aromatic scoring.
            aromatic_cutoff: Max distance for aromatic hotspot matching.
            n_aromatic_targets: Number of top aromatic hotspots to target.
            hbond_sigma: Gaussian width for H-bond scoring.
            hbond_cutoff: Max distance for H-bond hotspot matching.
            aromatic_exp: Exponent for aromatic term. Use <1.0 to soften.
            hbond_exp: Exponent for H-bond term. Use <1.0 to soften.
            floor: Minimum reward to maintain gradient signal.
        """
        super().__init__()
        
        # Load hotspot data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.cluster_centers = data['cluster_centers']
        self.cluster_counts = data['cluster_counts']
        self.cluster_features = data['cluster_features']
        self.target_profile = data['target_profile']
        
        # Store parameters
        self.aromatic_sigma = aromatic_sigma
        self.aromatic_cutoff = aromatic_cutoff
        self.hbond_sigma = hbond_sigma
        self.hbond_cutoff = hbond_cutoff
        self.aromatic_exp = aromatic_exp
        self.hbond_exp = hbond_exp
        
        # Build RDKit feature factory
        self.fdef = AllChem.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        )
        
        # Setup aromatic targets (top N by consensus count)
        self._setup_aromatic_targets(n_aromatic_targets)
        
        # Setup H-bond clusters grouped by type
        self._setup_hbond_clusters()
        
        # Logging
        print(f"MultiFeatureReward initialised:")
        print(f"  Hotspot file: {pkl_path}")
        print(f"  Aromatic: sigma={aromatic_sigma}, cutoff={aromatic_cutoff}, "
              f"n_targets={n_aromatic_targets}")
        print(f"  H-bond: sigma={hbond_sigma}, cutoff={hbond_cutoff}")
        print(f"  Combination: aromatic^{aromatic_exp} * hbond^{hbond_exp}")
        print(f"  Aromatic norm factor: {self.aromatic_norm}")
        print(f"  H-bond types: {list(self._hbond_clusters.keys())}")
    
    def _setup_aromatic_targets(self, n_targets: int):
        """
        Extract top N aromatic hotspots by consensus count.
        
        Normalises by the single largest hotspot so that hitting
        one hotspot perfectly gives score ~1.0.
        """
        aromatic_indices = [
            i for i, feat in enumerate(self.cluster_features) 
            if feat == 'Aromatic'
        ]
        
        # Sort by count descending and take top N
        sorted_indices = sorted(
            aromatic_indices, 
            key=lambda i: self.cluster_counts[i], 
            reverse=True
        )
        self.aromatic_indices = sorted_indices[:n_targets]
        
        self.aromatic_centers = self.cluster_centers[self.aromatic_indices]
        self.aromatic_counts = [self.cluster_counts[i] for i in self.aromatic_indices]
        
        # Normalise by max single hotspot (allows score=1.0 by hitting one well)
        self.aromatic_norm = max(self.aromatic_counts) if self.aromatic_counts else 1.0
    
    def _setup_hbond_clusters(self):
        """
        Group H-bond related clusters by feature type.
        
        Only includes Acceptor, Donor, and NegIonizable features.
        Sorted by consensus count for Hungarian matching priority.
        """
        self._hbond_clusters = {}
        
        for i, feat in enumerate(self.cluster_features):
            if feat in self.HBOND_WEIGHTS:
                if feat not in self._hbond_clusters:
                    self._hbond_clusters[feat] = []
                self._hbond_clusters[feat].append((
                    self.cluster_centers[i],
                    self.cluster_counts[i]
                ))
        
        # Sort each type by count descending
        for feat_type in self._hbond_clusters:
            self._hbond_clusters[feat_type] = sorted(
                self._hbond_clusters[feat_type],
                key=lambda x: x[1],
                reverse=True
            )
    
    @property
    def name(self) -> str:
        return "multi_features"
    
    def _score_aromatic(self, mol: Mol) -> float:
        """
        Score aromatic feature placement using Hungarian matching.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Normalised aromatic score between 0.0 and 1.0.
        """
        if mol is None or not self.aromatic_indices:
            return 0.0
        
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
            mol_aromatics = [f for f in raw_feats if f.GetFamily() == 'Aromatic']
        except Exception:
            return 0.0
        
        if not mol_aromatics:
            return 0.0
        
        n_mol = len(mol_aromatics)
        n_target = len(self.aromatic_centers)
        cost_matrix = np.zeros((n_mol, n_target))
        
        for r, feat in enumerate(mol_aromatics):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
            for c in range(n_target):
                dist = np.linalg.norm(pos - self.aromatic_centers[c])
                if dist <= self.aromatic_cutoff:
                    gaussian = np.exp(-0.5 * (dist / self.aromatic_sigma) ** 2)
                    cost_matrix[r, c] = -(gaussian * self.aromatic_counts[c])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_points = -cost_matrix[row_ind, col_ind].sum()
        
        score = total_points / self.aromatic_norm if self.aromatic_norm > 0 else 0.0
        return float(min(1.0, score))
    
    def _score_hbond_category(self, mol_feats: List, feat_type: str) -> float:
        """
        Score a single H-bond feature category using Hungarian matching.
        
        Args:
            mol_feats: List of RDKit features of this type from the molecule.
            feat_type: Feature type (Acceptor, Donor, or NegIonizable).
            
        Returns:
            Normalised category score between 0.0 and 1.0.
        """
        if feat_type not in self._hbond_clusters or not mol_feats:
            return 0.0
        
        # Get ideal count from target profile
        ideal_count, _ = self.target_profile.get(feat_type, (0, 0))
        if ideal_count == 0:
            return 0.0
        
        # Take top N clusters where N = ideal count
        targets = self._hbond_clusters[feat_type][:ideal_count]
        target_centers = [t[0] for t in targets]
        target_counts = [t[1] for t in targets]
        
        # Normalise by sum of target counts
        norm_factor = sum(target_counts)
        if norm_factor == 0:
            return 0.0
        
        # Build cost matrix
        cost_matrix = np.zeros((len(mol_feats), len(targets)))
        
        for r, feat in enumerate(mol_feats):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
            for c, center in enumerate(target_centers):
                dist = np.linalg.norm(pos - center)
                if dist <= self.hbond_cutoff:
                    gaussian = np.exp(-0.5 * (dist / self.hbond_sigma) ** 2)
                    cost_matrix[r, c] = -(gaussian * target_counts[c])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_points = -cost_matrix[row_ind, col_ind].sum()
        
        return float(min(1.0, total_points / norm_factor))
    
    def _score_hbond(self, mol: Mol) -> float:
        """
        Score H-bond features as weighted sum of categories.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Weighted H-bond score between 0.0 and 1.0.
        """
        if mol is None:
            return 0.0
        
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
        except Exception:
            return 0.0
        
        final_score = 0.0
        
        for feat_type, weight in self.HBOND_WEIGHTS.items():
            mol_feats = [f for f in raw_feats if f.GetFamily() == feat_type]
            category_score = self._score_hbond_category(mol_feats, feat_type)
            final_score += weight * category_score
        
        return float(np.clip(final_score, 0.0, 1.0))
    
    def score_mol(self, mol: Mol) -> Tuple[float, float, float]:
        if mol is None:
            return 0.0, 0.0, 0.0
        
        aromatic_score = self._score_aromatic(mol)
        hbond_score = self._score_hbond(mol)
        
        aromatic_term = aromatic_score ** self.aromatic_exp
        hbond_term = hbond_score ** self.hbond_exp
        combined = aromatic_term * hbond_term
        
        return float(min(1.0, combined)), aromatic_score, hbond_score
    
    def __call__(self, molecules: List[Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculate multiplicative reward for a batch of molecules.
        
        Args:
            molecules: List of RDKit molecule objects.
            dataset_info: Optional dataset metadata (unused).
            **kwargs: Additional arguments (unused).
            
        Returns:
            Tensor of combined scores with shape (len(molecules),).
        """
        scores = []
        
        for mol in molecules:
            try:
                combined, _, _ = self.score_mol(mol)
                scores.append(combined)
            except Exception:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def score_batch_detailed(self, molecules: List[Mol]) -> Dict[str, torch.Tensor]:
        combined_scores = []
        aromatic_scores = []
        hbond_scores = []
        
        for mol in molecules:
            try:
                combined, aromatic, hbond = self.score_mol(mol)
            except Exception:
                combined, aromatic, hbond = 0.0, 0.0, 0.0
            
            combined_scores.append(combined)
            aromatic_scores.append(aromatic)
            hbond_scores.append(hbond)
        
        return {
            'combined': torch.tensor(combined_scores, dtype=torch.float32),
            'aromatic': torch.tensor(aromatic_scores, dtype=torch.float32),
            'hbond': torch.tensor(hbond_scores, dtype=torch.float32),
        }