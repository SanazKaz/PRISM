import os
import torch
import pandas as pd
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from src.prism.reward.scorer import BaseReward
import math


class SillyWalksReward(BaseReward):
    """
    Simple SillyWalks Reward (Pat Walters' Original)
    
    Scores molecules by fraction of novel fingerprint bits.
    Novel bit = fingerprint bit not present in ChEMBL reference set.
    
    Score calculation:
    - novelty = (number of novel bits) / (total bits in molecule)
    - reward = 1.0 - novelty  (inverted so lower novelty = higher reward)
    
    Args:
        reference_path: Path to reference SMILES file (space-separated: SMILES Name)
        radius: Morgan fingerprint radius (default 2)
    """
    def __init__(self, reference_path: str, radius: int = 2):
        self.radius = radius
        self.count_dict = {}  # Dictionary of known bits from reference
        
        if not os.path.exists(reference_path):
            print(f"[!] SillyWalks reference not found: {reference_path}")
            return

        print(f"Loading SillyWalks reference from {reference_path}")
        df = pd.read_csv(reference_path, sep=" ", names=["SMILES", "Name"])
        
        # Build dictionary of all bits seen in reference molecules
        for smi in df["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, self.radius)
                for bit, count in fp.GetNonzeroElements().items():
                    self.count_dict[bit] = self.count_dict.get(bit, 0) + count
        
        print(f"Loaded {len(self.count_dict)} unique bits from {len(df)} molecules")

    @property
    def name(self) -> str:
        return "silly_walks"

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Calculate reward for each molecule.
        
        Returns:
            Tensor of rewards where higher = more common chemistry (less novel)
        """
        scores = []
        for mol in molecules:
            if mol is None:
                scores.append(0.0)
                continue
            
            fp = AllChem.GetMorganFingerprint(mol, self.radius)
            on_bits = list(fp.GetNonzeroElements().keys())
            
            if not on_bits:
                scores.append(0.0)
                continue
            
            # Count how many bits are NOT in the reference (novel)
            novel_bits = [bit for bit in on_bits if bit not in self.count_dict]
            novelty = len(novel_bits) / len(on_bits)
            print(f"Novelty: {novelty}")
            
            # Invert: lower novelty = higher reward
            reward = math.exp(-2.0 * novelty)
            scores.append(float(reward))
        
        return torch.tensor(scores, dtype=torch.float32)