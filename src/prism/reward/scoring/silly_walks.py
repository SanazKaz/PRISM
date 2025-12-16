# adapted from https://github.com/PatWalters/silly_walks by Pat Walters for PRISM
# data for smiles in reward_data folder under chembl_drugs.smi, and chembl_ring_systems.csv

import torch
import pandas as pd
import os
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from src.prism.reward.scorer import BaseReward

class SillyWalksReward(BaseReward):
    def __init__(self, reference_path: str, radius: int = 2):
        """
        Args:
            reference_path: Path to chembl_drugs.smi (Space delimited: SMILES Name)
            radius: Morgan fingerprint radius (default 2, same as ECFP4)
        """
        self.radius = radius
        # We use a set for O(1) lookups. 
        # The original script counted occurrences, but for scoring it only checks existence.
        self.safe_bits = set()
        
        if not os.path.exists(reference_path):
            print(f"[!] Warning: SillyWalks reference not found at {reference_path}")
        else:
            print(f"Loading SillyWalks reference from {reference_path}...")
            try:
                # Load ChemBL drugs (SMILES column usually comes first)
                # Using chunksize is safer for memory if the file is huge, 
                # but chembl_drugs.smi is usually small (~2-3k molecules).
                df = pd.read_csv(reference_path, sep=" ", names=["SMILES", "Name"])
                
                count = 0
                for smi in df["SMILES"]:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        # Use GetMorganFingerprint (Sparse/Count-based) to match Pat Walters' logic exactly
                        fp = AllChem.GetMorganFingerprint(mol, self.radius)
                        # Store the keys (bits) that are considered "safe"
                        for k in fp.GetNonzeroElements().keys():
                            self.safe_bits.add(k)
                        count += 1
                        
                print(f"  > Processed {count} reference drugs.")
                print(f"  > Learned {len(self.safe_bits)} safe substructures.")
                
            except Exception as e:
                print(f"[!] Error loading SillyWalks data: {e}")

    @property
    def name(self) -> str:
        return "silly_walks"

    def score_mol(self, mol: Chem.Mol) -> float:
        """
        Calculates Silly Walks Score.
        Original Silliness = (Bits NOT in Ref) / (Total Bits)
        RL Reward = 1.0 - Silliness
        """
        if mol is None:
            return 0.0
            
        try:
            # Generate fingerprint for the input molecule
            fp = AllChem.GetMorganFingerprint(mol, self.radius)
            on_bits = list(fp.GetNonzeroElements().keys())
            
            if not on_bits:
                return 0.0
            
            # Count how many bits in this molecule are "Silly" (never seen in reference)
            silly_bits = [bit for bit in on_bits if bit not in self.safe_bits]
            
            # Calculate the ratio (0.0 to 1.0)
            silliness_score = len(silly_bits) / len(on_bits)
            
            # INVERT for RL: We want to Maximize Safety, not Silliness
            return float(1.0 - silliness_score)
            
        except Exception:
            return 0.0

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        for mol in molecules:
            score = self.score_mol(mol)
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32)