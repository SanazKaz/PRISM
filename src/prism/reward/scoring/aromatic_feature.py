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
    Focused Reward targeting the 3 primary Aromatic Hotspots in the AmpC pocket.
    Used as the first stage of a hierarchical alignment strategy.
    """
    def __init__(self, pkl_path: str, 
                 sigma: float = 1.0, 
                 cutoff: float = 3.5,
                 n_aromatic_targets: int = 3,
                 use_curriculum: bool = True):
        super().__init__()
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Load hotspot data
        self.cluster_centers = data['cluster_centers']
        self.cluster_counts = data['cluster_counts']
        self.cluster_features = data['cluster_features']
        
        # Filter for Aromatic clusters only and take top N
        aromatic_indices = [i for i, feat in enumerate(self.cluster_features) if feat == 'Aromatic']
        
        # Sort by cluster count descending to find the "loudest" hotspots
        sorted_aromatic = sorted(aromatic_indices, key=lambda i: self.cluster_counts[i], reverse=True)
        self.target_indices = sorted_aromatic[:n_aromatic_targets]
        
        self.target_centers = self.cluster_centers[self.target_indices]
        self.target_counts = [self.cluster_counts[i] for i in self.target_indices]
        self.max_possible_score = sum(self.target_counts)
        
        # Curriculum setup
        self.use_curriculum = use_curriculum
        self.curriculum_start_epoch = 58
        self.curriculum_end_epoch = 158
        
        # Parameters to be updated by curriculum
        self.start_cutoff = 5.0
        self.target_cutoff = 2.5
        self.start_sigma = 1.5
        self.target_sigma = 0.8
        
        self.sigma = self.start_sigma if use_curriculum else sigma
        self.cutoff = self.start_cutoff if use_curriculum else cutoff
        
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        
        print(f"AromaticAnchorReward initialized with {len(self.target_indices)} hotspots.")
        print(f"Target Centers:\n{self.target_centers}")

    @property
    def name(self) -> str:
        return "aromatic_anchor"

    def update_epoch(self, epoch: int) -> None:
        if not self.use_curriculum:
            return
        
        progress = np.clip((epoch - self.curriculum_start_epoch) / 
                           (self.curriculum_end_epoch - self.curriculum_start_epoch), 0, 1)
        
        self.cutoff = self.start_cutoff + progress * (self.target_cutoff - self.start_cutoff)
        self.sigma = self.start_sigma + progress * (self.target_sigma - self.start_sigma)

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0
        
        try:
            # Only extract Aromatic features
            raw_feats = self.fdef.GetFeaturesForMol(mol)
            mol_aromatic_feats = [f for f in raw_feats if f.GetFamily() == 'Aromatic']
        except Exception:
            return 0.0
        
        if not mol_aromatic_feats:
            return 0.0
        
        # Build cost matrix: Molecule features (rows) x Target hotspots (cols)
        n_mol = len(mol_aromatic_feats)
        n_target = len(self.target_centers)
        cost_matrix = np.zeros((n_mol, n_target))
        
        for r, feat in enumerate(mol_aromatic_feats):
            pos = np.array([feat.GetPos().x, feat.GetPos().y, feat.GetPos().z])
            for c in range(n_target):
                dist = np.linalg.norm(pos - self.target_centers[c])
                if dist <= self.cutoff:
                    gaussian = np.exp(-0.5 * (dist / self.sigma) ** 2)
                    # Minimize negative score
                    cost_matrix[r, c] = -(gaussian * self.target_counts[c])
        
        # Hungarian matching for optimal alignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_score = -cost_matrix[row_ind, col_ind].sum()
        
        # Normalize by the max possible score (sum of counts of the 3 hotspots)
        final_score = total_score / self.max_possible_score if self.max_possible_score > 0 else 0.0
        
        return float(np.clip(final_score, 0.0, 1.0))

    def __call__(self, molecules: List[Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            scores.append(self.score_mol(mol))
        return torch.tensor(scores, dtype=torch.float32)