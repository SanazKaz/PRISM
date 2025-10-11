# src/prism/rewards/rewards.py

import torch
from abc import ABC, abstractmethod
from rdkit import Chem
from utils import batch_to_list
from src.models.diffsbdd.analysis.molecule_builder import build_molecule, process_molecule
import numpy as np


class BaseReward(ABC):
    """
    An abstract base class for all reward functions.
    This acts as a template, ensuring all reward classes have a consistent interface.
    """
    @abstractmethod
    def calculate(self, molecules: list, **kwargs) -> torch.Tensor:
        """
        All reward classes must implement this method.
        It should take a list of molecules and return a tensor of scores.
        """
        raise NotImplementedError

# --- Dummy Reward Class for Testing ---

class DummyMedChemReward:
    """
    A placeholder reward class that builds real molecules, prints SMILES,
    and calculates oxygen-based rewards.
    """
    def __init__(self, dataset_info=None, ddpm_module=None, **kwargs):
        print("✅ Initialized DummyMedChemReward with molecule building and oxygen reward.")
        self.dataset_info = dataset_info
        self.ddpm_module = ddpm_module
        self.w_oxygen = kwargs.get('w_oxygen', 1.0)

    def build_molecules_from_batch(self, xh_lig, lig_mask):
        """
        Build RDKit molecule objects from batched ligand tensors.
        """
        if hasattr(self, 'ddpm_module') and hasattr(self.ddpm_module, 'virtual_nodes') and self.ddpm_module.virtual_nodes:
            atom_types = xh_lig[:, 3:].argmax(1)
            vnode_mask = (atom_types == self.ddpm_module.virtual_atom)
            xh_lig = xh_lig[~vnode_mask]
            lig_mask = lig_mask[~vnode_mask]
            
            if xh_lig.shape[0] == 0:
                return [], {}
        
        x = xh_lig[:, :3].detach().cpu()
        atom_type = torch.argmax(xh_lig[:, 3:], dim=1).detach().cpu()
        lig_mask = lig_mask.cpu()

        molecules = []
        molecule_to_batch_idx = {}

        for batch_idx, mol_pc in enumerate(zip(batch_to_list(x, lig_mask),
                                            batch_to_list(atom_type, lig_mask))):
            try:
                mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
                mol = process_molecule(
                    mol,
                    add_hydrogens=False,
                    sanitize=True,
                    relax_iter=0,
                    largest_frag=True
                )
                if mol is not None:
                    molecules.append(mol)
                    molecule_to_batch_idx[len(molecules)-1] = batch_idx
            except Exception as e:
                print(f"Failed to build molecule for batch index {batch_idx}: {str(e)}")
                continue

        return molecules, molecule_to_batch_idx
    
    def oxygen_reward(self, mol: Chem.Mol) -> float:
        """
        Reward based on oxygen percentage of heavy atoms (squared).
        
        Returns:
            float: Squared ratio of oxygen atoms to total heavy atoms (0 to 1)
        """
        if mol is None:
            return 0.0
        
        try:
            # Count oxygen atoms
            num_oxygens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
            
            # Count total heavy atoms (non-hydrogen)
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            
            if num_heavy_atoms == 0:
                return 0.0
            
            # Calculate percentage as ratio
            oxygen_percentage = num_oxygens / num_heavy_atoms
            
            # Square it (makes smaller values even smaller, larger values closer to 1)
            square_root_reward = np.sqrt(oxygen_percentage)
            
            return square_root_reward
            
        except Exception as e:
            print(f"Error in oxygen reward: {e}")
            return 0.0

    def composite_reward(self, xh_lig, xh_pocket, global_lig_mask, 
                         global_pocket_mask, current_epoch=None, names=None):
        """
        Builds molecules, prints SMILES, and calculates oxygen rewards.
        
        Returns:
            tuple: A tuple of (rewards, raw_scores) as tensors.
        """
        if global_lig_mask.numel() == 0:
            print("WARNING: Dummy reward received empty masks, returning empty tensors.")
            return torch.tensor([]), torch.tensor([])

        device = xh_lig.device
        num_molecules = len(torch.unique(global_lig_mask))
        
        # Build molecules
        molecules, mol_to_batch_idx = self.build_molecules_from_batch(xh_lig, global_lig_mask)
        
        rewards = torch.full((num_molecules,), -0.1, device=device)
        raw_scores = torch.full((num_molecules,), -0.1, device=device)
        
        if not molecules:
            print(f"No valid molecules built from {num_molecules} attempts.")
            return rewards, raw_scores
        
        print(f"\n{'='*70}")
        print(f"DummyMedChemReward: {len(molecules)} valid molecules (Epoch {current_epoch})")
        print(f"{'='*70}")
        
        for local_idx, mol in enumerate(molecules):
            batch_idx = mol_to_batch_idx[local_idx]
            
            # Calculate oxygen reward (already squared inside the function)
            oxygen = self.oxygen_reward(mol)
            
            reward = self.w_oxygen * oxygen
            
            rewards[batch_idx] = reward
            raw_scores[batch_idx] = oxygen
            
            # Print SMILES and breakdown
            smiles = Chem.MolToSmiles(mol)
            num_atoms = mol.GetNumHeavyAtoms()
            num_oxygens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
            oxygen_pct = (num_oxygens / num_atoms * 100) if num_atoms > 0 else 0
            
            print(f"Mol {local_idx:2d}: {smiles}")
            print(f"         Heavy atoms: {num_atoms} | O: {num_oxygens} ({oxygen_pct:.1f}%) | O²_reward: {oxygen:.3f} | Total: {reward:.3f}")
        
        print(f"{'='*70}\n")
        
        return rewards, raw_scores