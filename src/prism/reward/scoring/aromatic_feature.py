import os
import pickle
import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from src.prism.reward.scorer import BaseReward

class AromaticFeatureReward(BaseReward):
    """
    Reward targeting the 3 primary Aromatic Hotspots in the AmpC pocket.
    Normalized so that perfectly hitting ANY ONE hotspot yields a score of ~1.0.
    """
    def __init__(self, pkl_path: str, 
                 sigma: float = 0.8, 
                 cutoff: float = 2.5,
                 n_aromatic_targets: int = 3):
        super().__init__()
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Load hotspot data
        self.cluster_centers = data['cluster_centers']
        self.cluster_counts = data['cluster_counts']
        self.cluster_features = data['cluster_features']
        
        # Filter for Aromatic clusters only and take top N
        aromatic_indices = [i for i, feat in enumerate(self.cluster_features) if feat == 'Aromatic']
        sorted_aromatic = sorted(aromatic_indices, key=lambda i: self.cluster_counts[i], reverse=True)
        self.target_indices = sorted_aromatic[:n_aromatic_targets]
        
        self.target_centers = self.cluster_centers[self.target_indices]
        self.target_counts = [self.cluster_counts[i] for i in self.target_indices]
        
        # NEW LOGIC: Normalize by the SINGLE largest hotspot count
        # This allows hitting just one hotspot to reach a score of 1.0.
        self.norm_factor = max(self.target_counts)
        
        self.sigma = sigma
        self.cutoff = cutoff
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        print(f"AromaticFeatureReward (HIT-ANY) initialized.")
        print(f"Parameters: sigma={self.sigma}, cutoff={self.cutoff}")
        print(f"Normalization factor (Max single hotspot): {self.norm_factor}")

    @property
    def name(self) -> str:
        return "aromatic_anchor"

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        
        try:
            raw_feats = self.fdef.GetFeaturesForMol(mol)
            mol_aromatic_feats = [f for f in raw_feats if f.GetFamily() == 'Aromatic']
        except Exception:
            return 0.0
        
        if not mol_aromatic_feats:
            return 0.0
        
        n_mol = len(mol_aromatic_feats)
        n_target = len(self.target_centers)
        cost_matrix = np.zeros((n_mol, n_target))
        
        for r, feat in enumerate(mol_aromatic_feats):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
            for c in range(n_target):
                dist = np.linalg.norm(pos - self.target_centers[c])
                if dist <= self.cutoff:
                    gaussian = np.exp(-0.5 * (dist / self.sigma) ** 2)
                    cost_matrix[r, c] = -(gaussian * self.target_counts[c])
        
        # Optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_points = -cost_matrix[row_ind, col_ind].sum()
        
        # Scale score by the single best hotspot weight instead of the total sum
        final_score = total_points / self.norm_factor if self.norm_factor > 0 else 0.0
        
        # Cap at 1.0 so the model doesn't over-optimize by trying to hit all 3
        return float(min(1.0, final_score))

    def __call__(self, molecules: List[Mol], **kwargs) -> torch.Tensor:
        scores = [self.score_mol(m) for m in molecules]
        return torch.tensor(scores, dtype=torch.float32)