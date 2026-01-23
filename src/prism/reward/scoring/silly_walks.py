import os
import torch
import pandas as pd
import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from src.prism.reward.scorer import BaseReward

# --- Part 1: Fragmentation Utility (Internal) ---

class RingSystemFinder:
    """Logic to isolate ring systems by cleaving non-ring single bonds."""
    def __init__(self):
        # Pattern for ring carbonyls/exocyclic double bonds we want to keep
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#18R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            if bnd:
                bnd.SetBoolProp("protected", True)

    def cleave_linker_bonds(self, mol):
        frag_bond_list = []
        for bnd in mol.GetBonds():
            # Cleave if: Not in ring, Not protected, and is a Single Bond
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())
        
        if frag_bond_list:
            return Chem.FragmentOnBonds(mol, frag_bond_list)
        return mol

    def cleanup_fragments(self, mol):
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_smiles_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                # Convert dummy atoms (*) to Hydrogens for standard SMILES
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        atm.SetAtomicNum(1)
                frag = Chem.RemoveAllHs(frag)
                ring_system_smiles_list.append(Chem.MolToSmiles(frag))
        return ring_system_smiles_list

    def find_ring_systems(self, mol):
        if mol is None: return []
        clone = Chem.Mol(mol)
        self.tag_bonds_to_preserve(clone)
        frag_mol = self.cleave_linker_bonds(clone)
        return self.cleanup_fragments(frag_mol)


# --- Part 2: PRISM Reward Classes ---

class SillyWalksReward(BaseReward):
    """
    Weighted Silliness Reward (Morgan Fingerprint Novelty)
    
    Uses frequency-weighted penalty with exponential scoring.
    - Common bits (seen often in reference): low/no penalty
    - Rare bits (seen few times): medium penalty  
    - Novel bits (never seen): high penalty
    
    Penalty formula: max(0, 5 - log10(count + 1))
    Score: exp(-total_penalty * scale)
    """
    def __init__(self, reference_path: str, radius: int = 2, penalty_scale: float = 2.0):
        """
        Args:
            reference_path: Path to reference SMILES file (space-separated: SMILES Name)
            radius: Morgan fingerprint radius (default 2)
            penalty_scale: Controls harshness of penalty (higher = harsher)
        """
        self.radius = radius
        self.scale = penalty_scale
        self.bit_counts = {}  # Store frequency of each bit
        
        if not os.path.exists(reference_path):
            print(f"[!] SillyWalks data not found: {reference_path}")
            return

        print(f"Loading SillyWalks (Bit Frequencies) from {reference_path}")
        df = pd.read_csv(reference_path, sep=" ", names=["SMILES", "Name"])
        
        # Count frequency of each bit across all reference molecules
        for smi in df["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, self.radius)
                for bit in fp.GetNonzeroElements().keys():
                    self.bit_counts[bit] = self.bit_counts.get(bit, 0) + 1
        
        print(f"Loaded {len(self.bit_counts)} unique bits from {len(df)} molecules")

    @property
    def name(self) -> str:
        return "silly_walks"

    def _bit_penalty(self, bit: int) -> float:
        """
        Calculate penalty for a single bit based on its frequency.
        
        - count = 100,000 -> penalty = 0.0 (very common)
        - count = 10,000  -> penalty = 1.0
        - count = 100     -> penalty = 3.0
        - count = 5       -> penalty = 4.2
        - count = 0       -> penalty = 5.0 (never seen)
        """
        count = self.bit_counts.get(bit, 0)
        penalty = max(0.0, 5.0 - np.log10(count + 1))
        return penalty

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
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
            
            # Sum penalties for all bits
            total_penalty = sum(self._bit_penalty(bit) for bit in on_bits)
            print(f"Molecule has {len(on_bits)} bits, total penalty: {total_penalty:.1f}, avg per bit: {total_penalty/len(on_bits):.2f}")
            avg_penalty = total_penalty / len(on_bits)
            effective_penalty = max(0.0, avg_penalty - 2.5)
            
            # Exponential decay: 0 penalty -> 1.0, more penalty -> lower score
            score = np.exp(-effective_penalty * self.scale)
            scores.append(float(score))
            
        
        print(f"SillyWalks scores: {scores}")
        return torch.tensor(scores, dtype=torch.float32)


class SillyRingsReward(BaseReward):
    """Soft Ring Silliness (Frequency-based normalization)"""
    def __init__(self, reference_path: str):
        self.finder = RingSystemFinder()
        self.ring_dict = {}
        
        if not os.path.exists(reference_path):
            print(f"[!] SillyRings data not found: {reference_path}")
            return

        print(f"Loading SillyRings (Counts) from {reference_path}")
        df = pd.read_csv(reference_path)
        # Using column indices to be safe against different header names
        # Expected: index 0 = ID, 1 = ring_system, 2 = count
        self.ring_dict = dict(zip(df.iloc[:, 1], df.iloc[:, 2]))

    @property
    def name(self) -> str:
        return "silly_rings"

    def score_mol(self, mol: Chem.Mol) -> float:
        """
        Calculates a soft ring silliness score with hard penalties for:
        1. No rings present
        2. No aromatic atoms present
        3. Rings not found in the reference ChEMBL dictionary
        changed to FORCE ring inclusion else return 0.0
        """
        if mol is None: 
            return 0.0
        
        # 1. Identify Ring Systems using the Finder utility
        rings = self.finder.find_ring_systems(mol)
        
        # Clause A: Hard zero if no rings are present at all
        if not rings: 
            return -1.0

        # 2. Check for Aromaticity
        # Clause B: Hard zero if no aromatic atoms are in the molecule
        # This ensures the 'SillyRings' reward strictly favors aromatic scaffolds
        has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if not has_aromatic:
            return -0.5

        # 3. Frequency Scoring for the 'Weakest Link' (rarest ring system)
        counts = [self.ring_dict.get(r, 0) for r in rings]
        min_freq = min(counts)
        
        
        # Clause C: Hard zero if the ring is entirely unknown (count of 0)
        if min_freq == 0:
            return 0.0
            
        # 4. Soft Scoring: Log-normalize the count, but no safety floor
        # Math: log10(1) = 0.0; log10(100,000) = 5.0
        # Dividing by 5.0 scales common rings toward 1.0
        score = (np.log10(min_freq)) / 5.0
        
        # Clip only the upper bound to ensure 1.0 is the maximum
        return float(np.clip(score, 0.0, 1.0))

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        scores = [self.score_mol(m) for m in molecules]
        return torch.tensor(scores, dtype=torch.float32)