"""Module to calculate SuCOS score."""

from __future__ import annotations

import os
import torch
from typing import List

import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolChemicalFeatures import BuildFeatureFactory
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist

from src.prism.reward.scorer import BaseReward
from src.prism.utils import get_reference_ligand

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


def get_sucos_score(
    mol_reference: Mol,
    mol_probe: Mol,
    conf_id_reference: int = -1,
    conf_id_probe: int = -1,
    heavy_only: bool | None = True,
) -> float:
    """Calculate the SuCOS score between a reference ligand and a list of probe ligands."""

    # explicit or implicit hydrogens should be same for both molecules
    mol_reference = handle_hydrogens(mol_reference, heavy_only=heavy_only)
    mol_probe = handle_hydrogens(mol_probe, heavy_only=heavy_only)

    feature_map_score = get_feature_map_score(
        mol_small=mol_reference,
        mol_large=mol_probe,
        conf_id_small=conf_id_reference,
        conf_id_large=conf_id_probe,
    )
    protrusion_distance = ShapeProtrudeDist(
        mol1=mol_reference,
        mol2=mol_probe,
        confId1=conf_id_reference,
        confId2=conf_id_probe,
        allowReordering=False,
        vdwScale=0.8,
        ignoreHs=True,
    )
    shape_overlap = max(1 - protrusion_distance, 0)

    # if no features, base on shape alone
    if not np.isnan(feature_map_score):
        sucos_score = 0.5 * feature_map_score + 0.5 * shape_overlap
    else:
        sucos_score = shape_overlap

    sucos_score = min(max(0.0, sucos_score), 1.0)  # clipping
    return float(sucos_score)


# --- THE REWARD CLASS WRAPPER ---

class SuCOSReward(BaseReward):
    @property
    def name(self) -> str:
        return "sucos"

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        Calculates SuCOS scores for a batch of molecules against their ground truth ligands.
        
        Args:
            molecules: List of generated RDKit molecules.
            dataset_info: Dictionary containing 'datadir' path to find ground truth files.
            **kwargs: Must contain 'names' (list of sample IDs).
        """
        scores = []
        
        # 1. Get sample names (needed to find GT files)
        names = kwargs.get('names', [])
        
        # Validate inputs
        if not names or len(names) != len(molecules):
            # print(f"[SuCOS ERROR] Mismatch: {len(molecules)} molecules vs {len(names)} names.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        for i, (mol_pred, name) in enumerate(zip(molecules, names)):
            try:
                # 2. Load Ground Truth Molecule
                # utils.get_reference_ligand handles loading and centering
                mol_true = get_reference_ligand(name, dataset_info)
                
                if mol_true is None:
                    # Error already logged in utils.py
                    scores.append(0.0)
                    continue

                if mol_pred is None:
                    scores.append(0.0)
                    continue
                
                num_conf = mol_true.GetNumConformers()
                if num_conf == 0:
                    scores.append(0.0)
                    continue
                    
                # Calculate SuCOS against all ground truth conformers and take max
                current_scores = [
                    get_sucos_score(mol_true, mol_pred, conf_id_reference=c, heavy_only=True) 
                    for c in range(num_conf)
                ]
                
                best_score = max(current_scores)
                scores.append(best_score)

            except Exception as e:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)