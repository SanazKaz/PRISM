"""Module to calculate SuCOS score for generated molecules"""

from __future__ import annotations

import os
import torch
from typing import List
from pathlib import Path

import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.rdMolChemicalFeatures import BuildFeatureFactory
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist

from src.prism.reward.scorer import BaseReward
from src.prism.utils import center_ligand_on_com

# --- PRE-INITIALIZATION OF FEATURE FACTORY ---
FACTORY = BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
PARAMETERS = {k: FeatMaps.FeatMapParams() for k in FACTORY.GetFeatureFamilies()}
KEEP = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)

# helper function to calculate feature map score
def get_feature_map_score(
    mol_small: Mol,
    mol_large: Mol,
    conf_id_small: int = -1,
    conf_id_large: int = -1,
) -> float:
    """Calculate the feature map score between two molecules."""

    # list features
    features_small = [f for f in FACTORY.GetFeaturesForMol(mol_small, confId=conf_id_small) if f.GetFamily() in KEEP]
    features_large = [f for f in FACTORY.GetFeaturesForMol(mol_large, confId=conf_id_large) if f.GetFamily() in KEEP]

    # create feature map based on small molecule
    feature_map = FeatMaps.FeatMap(feats=features_small, weights=[1] * len(features_small), params=PARAMETERS)
    feature_map.scoreMode = FeatMaps.FeatMapScoreMode.Best  # type: ignore[misc]

    # score features of large molecule present in small molecule
    feature_score = feature_map.ScoreFeats(features_large)

    # normalize score
    normalization_constant = min(feature_map.GetNumFeatures(), len(features_large))
    if normalization_constant > 0:
        return float(feature_score / normalization_constant)

    return np.nan


def handle_hydrogens(mol: Mol, heavy_only: bool | None = True) -> Mol:
    """Remove, add or do not modify hydrogens in a molecule."""
    if heavy_only is None:
        return mol
    if heavy_only:
        try:
            return RemoveHs(mol)
        except:
            return mol
    return AddHs(mol, addCoords=True)


def get_sucos_score_optimized(
    ref_data: dict,
    mol_probe: Mol,
    conf_id_probe: int = -1,
    heavy_only: bool | None = True,
) -> float:
    """
    Fast SuCOS using precomputed reference data.
    Only processes the probe molecule.
    """
    # Process probe molecule once
    mol_probe = handle_hydrogens(mol_probe, heavy_only=heavy_only)
    
    # FEATURES: Extract from probe, score against precomputed ref map
    features_probe = [
        f for f in FACTORY.GetFeaturesForMol(mol_probe, confId=conf_id_probe) 
        if f.GetFamily() in KEEP
    ]
    
    feature_score = ref_data['feature_map'].ScoreFeats(features_probe)
    normalization = min(ref_data['num_features'], len(features_probe))
    
    if normalization > 0:
        feature_map_score = feature_score / normalization
    else:
        feature_map_score = np.nan
    
    # SHAPE: Compare probe against preprocessed reference
    protrusion_distance = ShapeProtrudeDist(
        mol1=ref_data['mol_shape_ready'],  # Already preprocessed
        mol2=mol_probe,
        confId1=-1,
        confId2=conf_id_probe,
        allowReordering=False,
        vdwScale=0.8,
        ignoreHs=True,
    )
    shape_overlap = max(1 - protrusion_distance, 0)
    
    # Combine
    if not np.isnan(feature_map_score):
        sucos_score = 0.5 * feature_map_score + 0.5 * shape_overlap
    else:
        sucos_score = shape_overlap
    
    return float(max(0.0, min(1.0, sucos_score)))


# --- THE REWARD CLASS WRAPPER ---

class SuCOSReward(BaseReward):
    
    def __init__(self, dataset_info):
        self.reference_data = []
        self.dataset_info = dataset_info
        
        data_root = Path(self.dataset_info['datadir']).parent
        sdf_dir = data_root / '02_preprocessed' / 'sdf_files'
                
        for sdf in sdf_dir.glob('*.sdf'):
            mol = center_ligand_on_com(str(sdf))  # Make sure it's a string
            
            if mol is None:
                continue
            
            # DEBUG: Check if centering worked
            coords = mol.GetConformer(0).GetPositions()
            com = np.mean(coords, axis=0)
            
            # Preprocess for shape comparison
            mol_shape_ready = handle_hydrogens(mol, heavy_only=True)
            
            # Precompute features
            features = [f for f in FACTORY.GetFeaturesForMol(mol) if f.GetFamily() in KEEP]
            feature_map = FeatMaps.FeatMap(feats=features, weights=[1] * len(features), params=PARAMETERS)
            feature_map.scoreMode = FeatMaps.FeatMapScoreMode.Best
            
            self.reference_data.append({
                'mol_original': mol,
                'mol_shape_ready': mol_shape_ready,
                'features': features,
                'feature_map': feature_map,
                'num_features': len(features)
            })
        
    @property
    def name(self) -> str:
        return "sucos"

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        
        for idx, mol_pred in enumerate(molecules):
            try:
                if mol_pred is None:
                    scores.append(0.0)
                    continue
                
                # DEBUG: Check generated molecule position
                gen_coords = mol_pred.GetConformer(0).GetPositions()
                gen_com = np.mean(gen_coords, axis=0)
                
                # Compare against all references
                all_scores = []
                for ref_data in self.reference_data:
                    score = get_sucos_score_optimized(ref_data, mol_pred, heavy_only=True)
                    all_scores.append(score)
                
                best_score = max(all_scores) if all_scores else 0.0
                scores.append(best_score)
                
            except Exception as e:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)