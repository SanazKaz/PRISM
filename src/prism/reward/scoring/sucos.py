"""SuCOSReward replacement
- Sample a single reference per batch (avoids moving goalpost)
- Harmonic denominator between probe and reference feature counts
- Matched fraction threshold to penalize fragments
- Small downweight for tiny molecules
- Keeps color/shape combination and FeatMaps usage
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import torch
import numpy as np
from rdkit import RDConfig, Chem, DataStructs
from rdkit.Chem import rdMolChemicalFeatures as rdmolcf
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolChemicalFeatures import BuildFeatureFactory
from rdkit.Chem.rdmolops import AddHs, RemoveHs
from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist

# If you have your BaseReward and center helper in your package, import them:
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

# --- TUNABLE HYPERPARAMS (easy to change) ---
MIN_MATCHED_FRAC = 0.35      # minimum fraction of reference features that must be matched
TINY_MOL_HEAVY_ATOMS = 10    # threshold to consider molecule "tiny" and downweight
TINY_MOL_SCALE = 0.2        # multiplier for tiny molecules
RANDOM_SEED = 0             # seed for deterministic sampling (set None to be random)


def handle_hydrogens(mol: Mol, heavy_only: bool | None = True) -> Mol:
    """Remove, add or do not modify hydrogens in a molecule."""
    if heavy_only is None:
        return mol
    if heavy_only:
        try:
            return RemoveHs(mol)
        except Exception:
            return mol
    try:
        return AddHs(mol, addCoords=True)
    except Exception:
        return mol


def get_sucos_score_optimized(
    ref_data: dict,
    mol_probe: Mol,
    conf_id_probe: int = -1,
    heavy_only: bool | None = True,
) -> float:
    """
    Fast SuCOS using precomputed reference data.
    - Uses harmonic denominator between probe and ref feature counts.
    - Applies a matched fraction check (matched_count / ref_num_features).
    - Returns a float in [0,1].
    """
    if mol_probe is None:
        return 0.0

    # Prepare probe (heavy-only coordinate handling)
    mol_probe = handle_hydrogens(mol_probe, heavy_only=heavy_only)

    # --- FEATURES: Extract from probe, score against precomputed ref map ---
    try:
        features_probe = [
            f for f in FACTORY.GetFeaturesForMol(mol_probe, confId=conf_id_probe)
            if f.GetFamily() in KEEP
        ]
    except Exception:
        features_probe = []

    # Score features against reference's precomputed feature map
    try:
        feature_score = ref_data["feature_map"].ScoreFeats(features_probe)
    except Exception:
        feature_score = 0.0

    # Estimate matched_count (approx). When weights==1, ScoreFeats ≈ matched_count.
    matched_count = int(round(feature_score)) if feature_score is not None else 0
    probe_num_features = len(features_probe)
    ref_num_features = ref_data.get("num_features", 0)

    # --- HARMONIC NORMALIZATION ---
    if probe_num_features > 0 and ref_num_features > 0:
        denom = 2.0 * ref_num_features * probe_num_features / (ref_num_features + probe_num_features)
        if denom > 0:
            feature_map_score = float(feature_score) / denom
        else:
            feature_map_score = 0.0
    else:
        feature_map_score = 0.0

    # Matched fraction check: penalize if probe matches too few of the reference's features
    matched_frac = (matched_count / float(ref_num_features)) if ref_num_features > 0 else 0.0
    if matched_frac < MIN_MATCHED_FRAC:
        # strong downweight for low matched fraction
        feature_map_score *= 0.2

    # --- SHAPE: Compare probe against preprocessed reference geometry ---
    try:
        protrusion_distance = ShapeProtrudeDist(
            mol1=ref_data["mol_shape_ready"],
            mol2=mol_probe,
            confId1=-1,
            confId2=conf_id_probe,
            allowReordering=False,
            vdwScale=0.8,
            ignoreHs=True,
        )
        shape_overlap = max(1.0 - protrusion_distance, 0.0)
    except Exception:
        shape_overlap = 0.0

    # Combine: 0.5 * feature + 0.5 * shape (classic SuCOS)
    sucos_score = 0.5 * feature_map_score + 0.5 * shape_overlap

    # Ensure bounds
    sucos_score = float(max(0.0, min(1.0, sucos_score)))
    return sucos_score


# --- THE REWARD CLASS WRAPPER ---
class SuCOSReward(BaseReward):
    def __init__(self, dataset_info, top_k: int = 1, rng_seed: int | None = RANDOM_SEED):
        """
        top_k kept for API compatibility but not used here (we sample a single reference).
        This reward samples one reference per batch/call and computes SuCOS only to that ref.
        """
        super().__init__()
        self.dataset_info = dataset_info
        self.top_k = top_k
        self.reference_data: List[dict] = []
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # Load references from preprocessed SDFs (same logic as your previous loader)
        data_root = Path(self.dataset_info["datadir"]).parent
        sdf_dir = data_root / "02_preprocessed" / "sdf_files"

        seen_pdbs = set()
        for sdf in sdf_dir.glob("*.sdf"):
            pdb_code = sdf.name.split("_")[0]
            if pdb_code in seen_pdbs:
                # keep single representative per PDB code (optional)
                continue

            mol = center_ligand_on_com(str(sdf))
            if mol is None:
                continue

            # Skip tiny solvent-like fragments
            if mol.GetNumHeavyAtoms() < 5:
                continue

            seen_pdbs.add(pdb_code)

            # Preprocess for shape comparison
            mol_shape_ready = handle_hydrogens(mol, heavy_only=True)

            # Precompute features
            features = [f for f in FACTORY.GetFeaturesForMol(mol) if f.GetFamily() in KEEP]
            if not features:
                continue

            feature_map = FeatMaps.FeatMap(feats=features, weights=[1] * len(features), params=PARAMETERS)
            feature_map.scoreMode = FeatMaps.FeatMapScoreMode.Best

            self.reference_data.append(
                {
                    "mol_original": mol,
                    "mol_shape_ready": mol_shape_ready,
                    "features": features,
                    "feature_map": feature_map,
                    "num_features": len(features),
                    "source": str(sdf),
                }
            )

        if len(self.reference_data) == 0:
            raise ValueError("SuCOSReward: no references loaded. Check dataset_info/datadir and sdf files.")

        print(f"SuCOSReward loaded {len(self.reference_data)} references. Using single-ref sampling per batch.")

    @property
    def name(self) -> str:
        return "sucos"

    def sample_reference_for_batch(self) -> dict:
        """Uniformly sample one reference dict for the current batch/episode."""
        return self.rng.choice(self.reference_data)

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        """
        For each call (batch), we sample one reference and compute SuCOS to that ref only.
        This makes the target stationary during the policy update and avoids the 'average over all refs' pathology.
        """
        scores: List[float] = []
        # Sample one reference for the whole batch / policy update step
        chosen_ref = self.sample_reference_for_batch()
        ref_num_features = chosen_ref.get("num_features", 0)

        for idx, mol_pred in enumerate(molecules):
            try:
                if mol_pred is None:
                    scores.append(0.0)
                    continue

                # Compute SuCOS to the chosen reference
                sucos_val = get_sucos_score_optimized(chosen_ref, mol_pred, heavy_only=True)

                # Downweight tiny molecules (cheap fragment gaming prevention)
                try:
                    n_heavy = mol_pred.GetNumHeavyAtoms()
                except Exception:
                    n_heavy = 0
                if n_heavy < TINY_MOL_HEAVY_ATOMS:
                    sucos_val *= TINY_MOL_SCALE

                # Clip to [0,1]
                sucos_val = float(max(0.0, min(1.0, sucos_val)))
                scores.append(sucos_val)
            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)
