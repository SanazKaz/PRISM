"""Lilly MedChem Rules scoring function.
https://pubs.acs.org/doi/10.1021/jm301008n
This implementation is based on blog post:
https://iwatobipen.wordpress.com/2023/12/17/useful-package-for-filtering-molecules-of-python-rdkit-python-memo/ 
"""

import numpy as np
from rdkit import Chem
from medchem.structural.lilly_demerits import LillyDemeritsFilters
from .transformations import reverse_sigmoid


def lilly_medchem_score(
    mol: Chem.Mol,
    demerit_threshold: float = 160.0,
    min_atoms: int = 15,
    max_atoms: int = 50,
    max_size_rings: int = 6,
    min_num_rings: int = 1,
    max_num_rings: int = 4,
    max_size_chain: int = 6,
    verbose: bool = True) -> float:
    
    """Run Lilly MedChem Rules on a molecule and return normalized score.
    
    Args:
        mol: RDKit molecule object
        demerit_threshold: Threshold for demerit score
        min_atoms: Minimum number of heavy atoms
        max_atoms: Maximum number of heavy atoms
        max_size_rings: Maximum size of individual rings
        min_num_rings: Minimum number of rings
        max_num_rings: Maximum number of rings
        max_size_chain: Maximum chain length
        verbose: Print diagnostic information
        
    Returns:
        Normalized score where 1.0 is best and 0.0 is worst
    """
    if mol is None:
        return 0.0
    
    try:
        mol_no_h = Chem.RemoveHs(mol)
        
        dfilters = LillyDemeritsFilters(
            dthresh=demerit_threshold,
            min_atoms=min_atoms,
            hard_max_atoms=max_atoms,
            max_size_rings=max_size_rings,
            min_num_rings=min_num_rings,
            max_num_rings=max_num_rings,
            max_size_chain=max_size_chain,
        )
        
        result = dfilters(mols=[mol_no_h])
        result_row = result.iloc[0]
        
        demerit_score = result_row['demerit_score']
        
        if np.isnan(demerit_score):
            demerit_score = demerit_threshold * 2
        
        soft_threshold = demerit_threshold / 2.0
        score = reverse_sigmoid(demerit_score, k=0.1, center=soft_threshold)
        
        if verbose:
            status = result_row['status']
            reasons = result_row['reasons']
            print(f"Status: {status}, Demerits: {demerit_score}, Reasons: {reasons}")
            print(f"Final score: {score:.3f}")
        
        return float(score)
        
    except Exception as e:
        print(f"Error in Lilly MedChem scoring: {e}")
        import traceback
        traceback.print_exc()
        return 0.0