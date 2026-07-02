# src/prism/reward/mol_prop_reward.py

"""Target Profile Reward – Ähnlichkeit zu einem Ziel-Property-Profil (z.B. ABL1)"""

from typing import List
import logging

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from src.prism.reward.scorer import BaseReward  

logger = logging.getLogger(__name__)


# =============================================================================
# Konstanten – unverändert aus deinem REINVENT-Code
# =============================================================================
PROPERTY_DENOMINATORS = {
    'MW': 500, 'AliR_C': 4, 'AroR_C': 3, 'ChiA_C': 6, 'SA': 6,
    'NHOH_C': 6, 'HetA_C': 10, 'RotB_C': 8, 'BriA_C': 2,
    'HBD_C': 5, 'HBA_C': 10, 'FusedR_C': 6, 'LogP_C': 5,
}

# Alle Ziel-Profile, je Target eins
TARGET_PROFILES = {
    'ABL1': {
        'MW': 0.8678131068863543, 'AliR_C': 0.1886354034643008,
        'AroR_C': 1.1643430502746093, 'ChiA_C': 0.07062385579495846,
        'SA': 0.709681734966906, 'NHOH_C': 0.340445007745388,
        'HetA_C': 0.8637515842839036, 'RotB_C': 0.6403147444021968,
        'BriA_C': 0.007182087029995775, 'HBD_C': 0.37558090409801437,
        'HBA_C': 0.5269539501478665, 'FusedR_C': 0.26770877341219546,
        'LogP_C': 0.8732651989860588,
    },
    'MTB': {
        'MW': 0.8111834883720932, 'AliR_C': 0.1744186046511628,
        'AroR_C': 0.9379844961240309, 'ChiA_C': 0.07364341085271317,
        'SA': 0.6480620155038759, 'NHOH_C': 0.2906976744186046,
        'HetA_C': 0.7930232558139535, 'RotB_C': 0.5319767441860465,
        'BriA_C': 0.0, 'HBD_C': 0.3441860465116279,
        'HBA_C': 0.4372093023255813, 'FusedR_C': 0.17441860465116277,
        'LogP_C': 0.6427676744186049,
    },
    'PARP1': {
        'MW': 0.8108418437336133, 'AliR_C': 0.33206607236497115,
        'AroR_C': 1.0253452193672434, 'ChiA_C': 0.045839888131445544,
        'SA': 0.7156397482957525, 'NHOH_C': 0.37371088970459715,
        'HetA_C': 0.7530152071316204, 'RotB_C': 0.521303093864709,
        'BriA_C': 0.009176717357105402, 'HBD_C': 0.39711588883062404,
        'HBA_C': 0.45980597797587835, 'FusedR_C': 0.4375109246635203,
        'LogP_C': 0.5492155369690617,
    },
    'DPP4': {
        'MW': 0.7921101383720932, 'AliR_C': 0.36809593023255816,
        'AroR_C': 0.6539728682170542, 'ChiA_C': 0.2999031007751938,
        'SA': 0.7157461240310078, 'NHOH_C': 0.3775193798449612,
        'HetA_C': 0.8146511627906976, 'RotB_C': 0.5868459302325582,
        'BriA_C': 0.0375, 'HBD_C': 0.29343023255813955,
        'HBA_C': 0.4805813953488372, 'FusedR_C': 0.19253875968992246,
        'LogP_C': 0.4378100808139537,
    },
}


# =============================================================================
# Help-function?!
# =============================================================================
def _calculate_sa_score(mol: Chem.Mol) -> float:
    try:
        from src.models.diffsbdd.analysis.SA_Score.sascorer import calculateScore
        return calculateScore(mol)
    except ImportError:
        logger.warning(
            "REAL SA-Score not available – using rough heuristic"
        )
        n_rings      = rdMolDescriptors.CalcNumRings(mol)
        n_stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        score = 2.0 + (n_rings * 0.5) + (n_stereo * 0.3) + (n_bridgehead * 0.5)
        return min(score, 10.0)


def _compute_properties(mol: Chem.Mol) -> dict | None:
    if mol is None:
        return None
    try:
        ring_info = mol.GetRingInfo()
        return {
            'MW':      Descriptors.MolWt(mol)                                          / PROPERTY_DENOMINATORS['MW'],
            'AliR_C':  Lipinski.NumAliphaticRings(mol)                                 / PROPERTY_DENOMINATORS['AliR_C'],
            'AroR_C':  Lipinski.NumAromaticRings(mol)                                  / PROPERTY_DENOMINATORS['AroR_C'],
            'ChiA_C':  len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))     / PROPERTY_DENOMINATORS['ChiA_C'],
            'SA':      _calculate_sa_score(mol)                                         / PROPERTY_DENOMINATORS['SA'],
            'NHOH_C':  Lipinski.NHOHCount(mol)                                         / PROPERTY_DENOMINATORS['NHOH_C'],
            'HetA_C':  sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1,6)) / PROPERTY_DENOMINATORS['HetA_C'],
            'RotB_C':  Descriptors.NumRotatableBonds(mol)                              / PROPERTY_DENOMINATORS['RotB_C'],
            'BriA_C':  rdMolDescriptors.CalcNumBridgeheadAtoms(mol)                   / PROPERTY_DENOMINATORS['BriA_C'],
            'HBD_C':   rdMolDescriptors.CalcNumHBD(mol)                               / PROPERTY_DENOMINATORS['HBD_C'],
            'HBA_C':   rdMolDescriptors.CalcNumHBA(mol)                               / PROPERTY_DENOMINATORS['HBA_C'],
            'FusedR_C': sum(1 for i in range(ring_info.NumRings())
                            if ring_info.IsRingFused(i))                               / PROPERTY_DENOMINATORS['FusedR_C'],
            'LogP_C':  Crippen.MolLogP(mol)                                            / PROPERTY_DENOMINATORS['LogP_C'],
        }
    except Exception:
        return None


# =============================================================================
# The Reward-class
# =============================================================================
class TargetProfileReward(BaseReward):
    """
    Belohnt Moleküle anhand ihrer Ähnlichkeit zu einem normalisierten
    Property-Profil (z.B. ABL1_human). Score ∈ (0, 1].
    """

    def __init__(self, target_profile: dict, sharpness: float = 5.0):
        self.target_profile = target_profile
        self.sharpness = sharpness
        logger.info(f"[TargetProfileReward] sharpness={self.sharpness}, "
                    f"{len(self.target_profile)} properties")

    @property
    def name(self) -> str:
        return "target_profile"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            props = _compute_properties(mol)
            if props is None:
                scores.append(0.0)
                continue
            sq_diffs = [(props[p] - t) ** 2 for p, t in self.target_profile.items()]
            dist = float(np.sqrt(np.mean(sq_diffs)))
            scores.append(float(np.exp(-dist * self.sharpness)))

        return torch.tensor(scores, dtype=torch.float32)