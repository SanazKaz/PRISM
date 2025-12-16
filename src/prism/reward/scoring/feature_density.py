"""HotspotReward
- DBSCAN-based pharmacophore hotspot scoring
- Precomputed clusters from reference dataset
- Gaussian distance weighting for continuous gradient
- Target profile matching to prevent feature spamming
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import torch
import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from src.prism.reward.scorer import BaseReward


class FeatureDensityReward(BaseReward):
    def __init__(self, pkl_path: str, sigma: float = 1.0, cutoff: float = 2.5, 
                 placement_weight: float = 0.7, profile_weight: float = 0.3):
        """
        Args:
            pkl_path: Path to pickled hotspot data.
            sigma: Width of gaussian. Set to 1.0 for broader gradients during training.
            cutoff: Max distance to score. Set to 2.5 to catch near-misses.
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
        self.placement_weight = placement_weight
        self.profile_weight = profile_weight
        
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        print(f"HotspotReward loaded: {len(self.cluster_centers)} clusters")
        print(f"  sigma={self.sigma}, cutoff={self.cutoff}")

    @property
    def name(self) -> str:
        return "feature_density"

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
            mol_feats = [f for f in raw_feats if f.GetFamily() in self.keep]
        except Exception:
            return 0.0
        
        if len(mol_feats) == 0:
            return 0.0
        
        feats_by_type = {}
        for feat in mol_feats:
            ft = feat.GetFamily()
            if ft not in feats_by_type:
                feats_by_type[ft] = []
            feats_by_type[ft].append(feat)
        
        score = 0.0
        max_score = 0.0
        hits = set()
        
        for feat_type, (ideal_count, tolerance) in self.target_profile.items():
            if ideal_count == 0:
                continue
            
            # Select top clusters for this feature type
            type_clusters = [(i, self.cluster_centers[i], self.cluster_counts[i]) 
                             for i in range(len(self.cluster_features)) 
                             if self.cluster_features[i] == feat_type]
            type_clusters = sorted(type_clusters, key=lambda x: x[2], reverse=True)[:ideal_count]
            
            # Calculate theoretical max score for normalization
            for i, center, count in type_clusters:
                max_score += count
            
            if feat_type not in feats_by_type:
                continue
            
            # Score the molecule's features against these clusters
            for feat in feats_by_type[feat_type]:
                pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
                
                for i, center, count in type_clusters:
                    if i in hits:
                        continue
                    
                    dist = np.linalg.norm(pos - center)
                    if dist <= self.cutoff:
                        gaussian = np.exp(-0.5 * (dist / self.sigma) ** 2)
                        score += gaussian * count
                        hits.add(i)
                        break
        
        placement_score = score / max_score if max_score > 0 else 0.0
        
        print(f"Placement score: {placement_score}")
        # Calculate Profile Score (Count matching)
        profile_score = 0.0
        for feat_type, (ideal_count, tolerance) in self.target_profile.items():
            actual_count = len(feats_by_type.get(feat_type, []))
            if ideal_count == 0:
                if actual_count == 0:
                    profile_score += 1.0
            else:
                profile_score += np.exp(-0.5 * ((actual_count - ideal_count) / tolerance) ** 2)
        profile_score /= len(self.target_profile)
        
        final_score = self.placement_weight * placement_score + self.profile_weight * profile_score
        
        return float(max(0.0, min(1.0, final_score)))

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            try:
                score = self.score_mol(mol)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)