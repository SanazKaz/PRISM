"""MMFF strain energy reward for molecular conformations."""

import math
import torch
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField, 
    MMFFHasAllMoleculeParams,
    MMFFGetMoleculeProperties,
    UFFGetMoleculeForceField,
    UFFHasAllMoleculeParams
)

from src.prism.reward.scorer import BaseReward


class MMFFStrainReward(BaseReward):
    """
    Reward based on MMFF strain energy of generated conformations.
    
    Strain = E_observed - E_minimized
    Lower strain means more chemically reasonable geometry.
    Transformed to 0-1 where 1 = no strain, 0 = highly strained.
    """
    
    def __init__(self, 
                 scale: float = 200.0,
                 max_iterations: int = 500,
                 debug: bool = True):
        """
        Args:
            scale: Controls steepness of exponential transformation.
                   Higher scale = more lenient (slower decay to 0).
            max_iterations: Maximum steps for MMFF minimisation.
            debug: Whether to print debug information.
        """
        self.scale = scale
        self.max_iterations = max_iterations
        self.debug = debug
    
    def _get_smiles_safe(self, mol: Chem.Mol) -> str:
        """Safely get SMILES string for logging."""
        try:
            return Chem.MolToSmiles(mol) if mol else "None"
        except:
            return "SMILES_ERROR"
    
    def _prepare_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Prepare molecule for energy calculation: sanitize and add hydrogens.
        
        Uses UFF to optimise hydrogen positions after adding them,
        similar to PoseBusters approach.
        
        Args:
            mol: Input RDKit Mol.
            
        Returns:
            Prepared molecule with explicit Hs, or None if preparation fails.
        """
        smiles = self._get_smiles_safe(mol)
        
        try:
            # Work on a copy
            mol_copy = Chem.Mol(mol)
            
            # Try to sanitize first
            try:
                Chem.SanitizeMol(mol_copy)
            except Exception as e:
                if self.debug:
                    print(f"[MMFF DEBUG] Sanitization failed for {smiles}: {e}")
                return None
            
            # Add hydrogens with coordinates
            mol_h = Chem.AddHs(mol_copy, addCoords=True)
            
            if mol_h.GetNumConformers() == 0:
                if self.debug:
                    print(f"[MMFF DEBUG] Lost conformer after AddHs for {smiles}")
                return None
            
            # Optimise just the hydrogen positions using UFF
            # This fixes poorly placed Hs from addCoords=True
            if UFFHasAllMoleculeParams(mol_h):
                ff = UFFGetMoleculeForceField(mol_h)
                if ff is not None:
                    # Fix all heavy atoms, only optimise hydrogens
                    for atom in mol_h.GetAtoms():
                        if atom.GetAtomicNum() != 1:  # Not hydrogen
                            ff.AddFixedPoint(atom.GetIdx())
                    ff.Minimize(maxIts=200)
            
            return mol_h
            
        except Exception as e:
            if self.debug:
                print(f"[MMFF DEBUG] Molecule preparation failed for {smiles}: {e}")
            return None
    
    def _calculate_strain(self, mol: Chem.Mol) -> Optional[float]:
        """
        Calculate strain energy: E_observed - E_minimized.
        
        Args:
            mol: RDKit Mol with at least one conformer.
            
        Returns:
            Strain energy in kcal/mol, or None if calculation fails.
        """
        # Check molecule and conformer exist
        if mol is None:
            if self.debug:
                print("[MMFF DEBUG] mol is None")
            return None
            
        if mol.GetNumConformers() == 0:
            if self.debug:
                smiles = self._get_smiles_safe(mol)
                print(f"[MMFF DEBUG] No conformer for: {smiles}")
            return None
        
        smiles = self._get_smiles_safe(mol)
        
        # Prepare molecule (sanitize + add Hs with proper positions)
        mol_prepared = self._prepare_molecule(mol)
        if mol_prepared is None:
            return None
        
        try:
            # Check MMFF parameters are available
            if not MMFFHasAllMoleculeParams(mol_prepared):
                if self.debug:
                    print(f"[MMFF DEBUG] No MMFF params for: {smiles}")
                return None
            
            # Get MMFF properties
            mmff_props = MMFFGetMoleculeProperties(mol_prepared)
            if mmff_props is None:
                if self.debug:
                    print(f"[MMFF DEBUG] MMFFGetMoleculeProperties failed for: {smiles}")
                return None
            
            # Get energy of observed (generated) conformation
            ff_observed = MMFFGetMoleculeForceField(mol_prepared, mmff_props, confId=0)
            if ff_observed is None:
                if self.debug:
                    print(f"[MMFF DEBUG] ForceField creation failed for: {smiles}")
                return None
            
            e_observed = ff_observed.CalcEnergy()
            
            # Create new force field for minimisation
            ff_minimise = MMFFGetMoleculeForceField(mol_prepared, mmff_props, confId=0)
            if ff_minimise is None:
                if self.debug:
                    print(f"[MMFF DEBUG] ForceField for minimisation failed for: {smiles}")
                return None
            
            # Minimise and get relaxed energy
            ff_minimise.Minimize(maxIts=self.max_iterations)
            e_minimised = ff_minimise.CalcEnergy()
            
            # Calculate strain
            strain = e_observed - e_minimised
            
            if self.debug:
                print(f"[MMFF DEBUG] Success: {smiles[:40]}... | "
                      f"E_obs={e_observed:.1f}, E_min={e_minimised:.1f}, strain={strain:.1f}")
            
            return max(0.0, strain)  # Clamp negative values
            
        except Exception as e:
            if self.debug:
                print(f"[MMFF DEBUG] Exception for {smiles}: {type(e).__name__}: {e}")
            return None
    
    def _transform_to_reward(self, strain: Optional[float]) -> float:
        """
        Transform strain energy to [0, 1] reward.
        
        Uses exponential decay: reward = exp(-strain / scale)
        - strain = 0 -> reward = 1.0
        - strain = scale -> reward approx 0.37
        - strain >> scale -> reward approx 0.0
        
        Args:
            strain: Strain energy in kcal/mol, or None for failed calcs.
            
        Returns:
            Reward value between 0 and 1.
        """
        if strain is None:
            return 0.0
        
        return math.exp(-strain / self.scale)

    @property
    def name(self) -> str:
        return "mmff_strain"
    
    def __call__(self, 
                 molecules: List[Chem.Mol], 
                 **kwargs) -> torch.Tensor:
        """
        Calculate strain-based reward for each molecule.
        
        Args:
            molecules: List of RDKit Mol objects with 3D conformers.
            **kwargs: Additional arguments (unused, for compatibility).
            
        Returns:
            Tensor of rewards in [0, 1], shape (len(molecules),).
        """
        rewards = []
        
        for mol in molecules:
            strain = self._calculate_strain(mol)
            reward = self._transform_to_reward(strain)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)