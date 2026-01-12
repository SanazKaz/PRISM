"""HotspotReward
- DBSCAN-based pharmacophore hotspot scoring
- Precomputed clusters from reference dataset
- Gaussian distance weighting for continuous gradient
- Per-feature-type scoring with configurable weights
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward


class FeatureDensityReward(BaseReward):
    
    # Default weights - prioritise aromatics, downweight easy hydrophobes
    DEFAULT_FEATURE_WEIGHTS = {
        'Aromatic': 0.35,
        'Acceptor': 0.25,
        'Donor': 0.20,
        'Hydrophobe': 0.05,
        'NegIonizable': 0.05,
        'PosIonizable': 0.00,
        'LumpedHydrophobe': 0.05,
        'ZnBinder': 0.05,
    }
    
    def __init__(self, pkl_path: str, sigma: float = 1.0, cutoff: float = 3.0,
                 feature_weights: Dict[str, float] = None,
                 aromatic_gate_threshold: float = 0.1,
                 aromatic_gate_penalty: float = 0.6):
        """
        Args:
            pkl_path: Path to pickled hotspot data.
            sigma: Width of gaussian. Set to 1.0 for broader gradients during training.
            cutoff: Max distance to score. Set to 2.5 to catch near-misses.
            feature_weights: Dict mapping feature types to their importance weights.
                             Weights should sum to 1.0. If None, uses DEFAULT_FEATURE_WEIGHTS.
            aromatic_gate_threshold: Minimum aromatic score to avoid penalty.
            aromatic_gate_penalty: Multiplier applied if aromatic score below threshold.
        """
        super().__init__()
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.cluster_centers = data['cluster_centers']
        self.cluster_counts = data['cluster_counts']
        self.cluster_features = data['cluster_features']
        self.target_profile = data['target_profile']
        self.keep = data['keep']
        self.metadata = data.get('metadata', {})
        
        self.sigma = sigma
        self.cutoff = cutoff
        
        # Feature weights - use provided or defaults
        self.feature_weights = feature_weights if feature_weights else self.DEFAULT_FEATURE_WEIGHTS.copy()
        
        # Aromatic gate parameters
        self.aromatic_gate_threshold = aromatic_gate_threshold
        self.aromatic_gate_penalty = aromatic_gate_penalty
        
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        # Precompute clusters grouped by feature type for efficiency
        self._clusters_by_type = self._group_clusters_by_type()
        
        print(f"HotspotReward loaded: {len(self.cluster_centers)} clusters")
        print(f"  sigma={self.sigma}, cutoff={self.cutoff}")
        print(f"  Feature weights: {self.feature_weights}")
        print(f"  Aromatic gate: threshold={aromatic_gate_threshold}, penalty={aromatic_gate_penalty}")

    def _group_clusters_by_type(self) -> Dict[str, List[Tuple[int, np.ndarray, int]]]:
        """
        Precompute clusters grouped by feature type, sorted by count descending.
        
        Returns:
            Dict mapping feature type to list of (index, center, count) tuples.
        """
        clusters_by_type = {}
        for i in range(len(self.cluster_features)):
            ft = self.cluster_features[i]
            if ft not in clusters_by_type:
                clusters_by_type[ft] = []
            clusters_by_type[ft].append((i, self.cluster_centers[i], self.cluster_counts[i]))
        
        # Sort each type by count descending
        for ft in clusters_by_type:
            clusters_by_type[ft] = sorted(clusters_by_type[ft], key=lambda x: x[2], reverse=True)
        
        return clusters_by_type

    @property
    def name(self) -> str:
        return "feature_density"

    def _score_feature_type(self, feat_type: str, mol_feats: List, 
                            ideal_count: int) -> Tuple[float, float]:
        """
        Score a single feature type using Hungarian matching.
        
        Args:
            feat_type: The pharmacophore feature type (e.g., 'Aromatic').
            mol_feats: List of RDKit feature objects of this type from the molecule.
            ideal_count: Target number of features from the profile.
            
        Returns:
            Tuple of (normalised_score, max_possible_score).
        """
        if feat_type not in self._clusters_by_type:
            return 0.0, 0.0
        
        # Take top N clusters where N = ideal count
        type_clusters = self._clusters_by_type[feat_type][:ideal_count]
        
        if not type_clusters:
            return 0.0, 0.0
        
        # Max possible score for this feature type (perfect placement at all clusters)
        max_score = sum(count for _, _, count in type_clusters)
        
        if not mol_feats:
            # Molecule has no features of this type
            return 0.0, max_score
        
        # Build cost matrix for Hungarian algorithm
        # Rows: molecule features, Cols: target clusters
        cost_matrix = np.zeros((len(mol_feats), len(type_clusters)))
        
        for r, feat in enumerate(mol_feats):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
            
            for c, (idx, center, count) in enumerate(type_clusters):
                # New (Linear):
                dist = np.linalg.norm(pos - center)
                if dist <= self.cutoff:
                    linear_score = 1 - (dist / self.cutoff) ** 2
                    cost_matrix[r, c] = -(linear_score * count)
                # else: leave as 0 (no contribution)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        score = -cost_matrix[row_ind, col_ind].sum()
        
        # Normalise to 0-1 for this feature type
        normalised = score / max_score if max_score > 0 else 0.0
        
        return normalised, max_score

    def score_mol(self, mol: Mol) -> float:
        """
        Score a molecule using per-feature-type placement with weighted combination.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Final score between 0 and 1.
        """
        if mol is None:
            return 0.0
        
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
            mol_feats = [f for f in raw_feats if f.GetFamily() in self.keep]
        except Exception:
            return 0.0
        
        # Group molecule's features by type
        feats_by_type = {}
        for feat in mol_feats:
            ft = feat.GetFamily()
            if ft not in feats_by_type:
                feats_by_type[ft] = []
            feats_by_type[ft].append(feat)
        
        # Score each feature type independently
        type_scores = {}
        for feat_type, (ideal_count, tolerance) in self.target_profile.items():
            if ideal_count == 0:
                continue
            
            mol_feats_of_type = feats_by_type.get(feat_type, [])
            score, max_possible = self._score_feature_type(feat_type, mol_feats_of_type, ideal_count)
            type_scores[feat_type] = score
        
        # Weighted combination - each feature type gets equal voice scaled by weight
        final_score = 0.0
        total_weight = 0.0
        
        for feat_type, score in type_scores.items():
            weight = self.feature_weights.get(feat_type, 0.0)
            final_score += weight * score
            total_weight += weight
        
        # Normalise by total weight used (in case some types aren't in profile)
        if total_weight > 0:
            final_score /= total_weight
        
        # Apply aromatic gate
        aromatic_score = type_scores.get('Aromatic', 0.0)
        if aromatic_score < self.aromatic_gate_threshold:
            final_score *= self.aromatic_gate_penalty
        
        # Debug printing
        print(f"Type scores: {', '.join([f'{k}={v:.3f}' for k, v in type_scores.items()])}")
        print(f"Aromatic score: {aromatic_score:.3f}, Final: {final_score:.3f}")
        
        return float(max(0.0, min(1.0, final_score)))

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculate scores for a batch of molecules.
        
        Args:
            molecules: List of RDKit molecule objects.
            dataset_info: Optional dataset metadata (unused but kept for interface).
            
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