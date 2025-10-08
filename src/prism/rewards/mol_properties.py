# src/prism/rewards/rewards.py

import torch
from abc import ABC, abstractmethod

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
    A placeholder reward class that mimics the interface of your real
    MedChemReward but returns random values. Perfect for testing the pipeline.
    """
    def __init__(self, **kwargs):
        print("✅ Initialized DummyMedChemReward for testing.")
        # This constructor can accept any arguments but won't use them.
        pass

    def composite_reward(self, xh_lig, xh_pocket, global_lig_mask, 
                         global_pocket_mask, current_epoch=None, names=None):
        """
        Calculates a random reward for each molecule in the batch.
        
        Returns:
            tuple: A tuple of (rewards, raw_scores) as tensors.
        """
        if global_lig_mask.numel() == 0:
            print("WARNING: Dummy reward received empty masks, returning empty tensors.")
            return torch.tensor([]), torch.tensor([])

        # Determine the number of unique molecules from the mask
        num_molecules = len(torch.unique(global_lig_mask))
        
        print(f"Dummy reward is generating {num_molecules} random scores.")

        # Generate random rewards and scores between 0 and 1
        dummy_rewards = torch.rand(num_molecules)
        dummy_raw_scores = torch.rand(num_molecules)
        
        # Ensure tensors are on the correct device
        device = xh_lig.device
        return dummy_rewards.to(device), dummy_raw_scores.to(device)