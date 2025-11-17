# scoring_functions/molecular_properties.py
"""2D molecular property scoring functions. 
Adapted from DiffSBDD's analysis module."""

from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from src.models.diffsbdd.analysis.SA_Score.sascorer import calculateScore


def qed_score(mol: Chem.Mol) -> float:
    """Calculate QED score (drug-likeness, 0-1)."""
    # custom weights for QED score to reduce rigidity weight
    # without reduction, heavily fused rings are generated 
    custom_weights = (0.66, 0.46, 0.05, 0.61, 0.06, 0.30, 0.48, 0.95)
    try:
        return float(QED.qed(mol, weights=custom_weights))
    except:
        return 0.0


def sa_score(mol: Chem.Mol) -> float:
    """Calculate normalized SA score (higher = easier to synthesize, 0-1)."""
    try:
        sa = calculateScore(mol)
        return round((10 - sa) / 9, 2)
    except:
        return 0.0


def logp_score(mol: Chem.Mol) -> float:
    """Calculate LogP (lipophilicity)."""
    try:
        return Crippen.MolLogP(mol)
    except:
        return 0.0


def lipinski_score(mol: Chem.Mol) -> float:
    """Calculate number of Lipinski rules satisfied (0-5)."""
    try:
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = Crippen.MolLogP(mol)
        rule_4 = (-2 <= logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        return sum([rule_1, rule_2, rule_3, rule_4, rule_5])
    except:
        return 0.0
    
    
def bertz_complexity_score(mol: Chem.Mol) -> float:
    """Calculate Bertz complexity score (0-1)."""
    try:
        return GraphDescriptors.BertzCT(mol)
    # TODO: Implement transform to be between 0 and 1
    except:
        return 0.0
    