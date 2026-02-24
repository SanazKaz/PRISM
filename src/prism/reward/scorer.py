import torch
import traceback
from typing import List, Dict, Tuple, Any, Optional
from rdkit import Chem
from abc import ABC, abstractmethod
from src.prism.reward.scoring.transformations import reshape_batch_rewards

# Import the utility we created previously
from src.prism.utils import build_molecules_from_batch

class BaseReward(ABC):
    """
    Interface that all specific reward classes (e.g., QED, Affinity) must implement.
    """
    @abstractmethod
    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Calculate scores for a list of molecules.
        Returns a tensor of shape (len(molecules),).
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the reward for logging purposes."""
        pass
    
    @property
    def epoch_weight_schedule(self) -> Optional[int]:
        """Epoch threshold for weight swap. None means no scheduling."""
        return None

    @property
    def weight_before_epoch(self) -> Optional[float]:
        """Weight to use before epoch threshold. None means use config weight."""
        return None
    
    @property
    def weight_after_epoch(self) -> Optional[float]:
        """Weight to use after epoch threshold. None means use config weight."""
        return None

    # TODO: remove from rewards later - keeping for backwards compatibility
    @property
    def increase_weight_after_epoch(self) -> Optional[int]:
        """The epoch after which to increase the weight.
        If None, the weight will not be increased.
        """
        return None
    
    @property
    def increased_weight_multiplier(self) -> Optional[float]:
        """Multiplier applied to the original weight after the epoch threshold.
        If None, the weight will not be increased.
        """
        return 1.0
        

class RewardManager:
    """
    The main orchestrator for PPO training. 
    
    Responsibilities:
    1. Reconstructs molecules from generative model tensors.
    2. Delegates scoring to a collection of specific reward instances.
    3. Aggregates weighted scores into a final reward tensor.
    """

    def __init__(self, 
                 reward_fns: List[BaseReward], 
                 reward_weights: Dict[str, float], 
                 dataset_info: Any, 
                 ddpm_module: Any = None):
        """
        Args:
            reward_fns: List of instantiated reward classes inheriting from BaseReward.
            reward_weights: Dictionary mapping reward names to their float weights.
            dataset_info: Metadata required for molecule reconstruction and file finding.
            ddpm_module: Optional module for virtual node handling.
        """
        self.reward_fns = reward_fns
        self.weights = reward_weights
        self.base_weights = reward_weights.copy()
        self.dataset_info = dataset_info
        self.ddpm_module = ddpm_module
        
        # Validate weights match rewards
        for reward in self.reward_fns:
            if reward.name not in self.weights:
                raise ValueError(f"Weight for reward '{reward.name}' not found in reward_weights.")

    def _get_effective_weight(self, reward_fn: BaseReward, current_epoch: int) -> float:
        base_weight = self.base_weights[reward_fn.name]

        if reward_fn.epoch_weight_schedule is not None:
            if current_epoch < reward_fn.epoch_weight_schedule:
                weight = reward_fn.weight_before_epoch if reward_fn.weight_before_epoch is not None else base_weight
            else:
                weight = reward_fn.weight_after_epoch if reward_fn.weight_after_epoch is not None else base_weight
            
            # Log the transition exactly once
            if current_epoch == reward_fn.epoch_weight_schedule:
                print(f"[RewardManager] Weight schedule triggered for '{reward_fn.name}': "
                    f"{reward_fn.weight_before_epoch} -> {reward_fn.weight_after_epoch} "
                    f"at epoch {current_epoch}")
            
            return weight

        if reward_fn.increase_weight_after_epoch is not None:
            if current_epoch >= reward_fn.increase_weight_after_epoch:
                return base_weight * reward_fn.increased_weight_multiplier

        return base_weight
    
    
    def __call__(self, 
                 xh_lig: torch.Tensor, 
                 global_lig_mask: torch.Tensor, 
                 current_epoch: int = 0,
                 **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the composite reward for a batch of ligand tensors.

        Args:
            xh_lig: Ligand features/coordinates tensor.
            global_lig_mask: Mask identifying which atom belongs to which molecule.
            current_epoch: Current training epoch (used for logging sparsity).
            **kwargs: Additional arguments passed to reward functions (e.g., 'names', 'xh_pocket').

        Returns:
            total_rewards: Tensor of shape (batch_size,) containing weighted sum.
            component_scores: Dictionary of raw scores for logging/debugging.
        """
        device = xh_lig.device

        # 1. Handle empty or invalid batches immediately
        if global_lig_mask.numel() == 0:
            return torch.tensor([], device=device), {}

        num_molecules = len(torch.unique(global_lig_mask))
        
        total_rewards = torch.full((num_molecules,), -0.1, device=device, dtype=torch.float32)
        
        component_scores = {
            r.name: torch.zeros(num_molecules, device=device, dtype=torch.float32) 
            for r in self.reward_fns
        }

        molecules, mol_to_batch_idx = build_molecules_from_batch(
            xh_lig, 
            global_lig_mask, 
            self.dataset_info, 
            self.ddpm_module
        )

        if not molecules:
            return total_rewards, component_scores

        valid_indices = list(mol_to_batch_idx.values())
        
        # Reset the valid slots to 0.0 before accumulating weighted sums
        total_rewards[valid_indices] = 0.0
        for reward_fn in self.reward_fns:
            try:
                # Calculate raw scores for the list of RDKit objects.
                raw_scores = reward_fn(
                    molecules, 
                    dataset_info=self.dataset_info, 
                    **kwargs
                )
                
                
                if not isinstance(raw_scores, torch.Tensor):
                    raw_scores = torch.tensor(raw_scores, device=device, dtype=torch.float32)
                else:
                    raw_scores = raw_scores.to(device).float()
                
                raw_scores = torch.nan_to_num(raw_scores, nan=0.0)
                
                weight = self._get_effective_weight(reward_fn, current_epoch)
                
                # Map local list indices back to the original batch tensor
                for local_idx, batch_idx in mol_to_batch_idx.items():
                    score = raw_scores[local_idx]
                    
                    # Accumulate weighted sum
                    total_rewards[batch_idx] += score * weight
                    
                    # Store raw component score
                    component_scores[reward_fn.name][batch_idx] = score

            except Exception as e:
                traceback.print_exc()

        
        print(f"\n[Epoch {current_epoch}] Molecule Rewards:")
        print(f"{'SMILES':<60} {'Total':<10} {' | '.join([r.name for r in self.reward_fns])}")
        print("-" * 110)
        for local_idx, mol in enumerate(molecules):
            batch_idx = mol_to_batch_idx[local_idx]
            smiles = Chem.MolToSmiles(mol)
            total = total_rewards[batch_idx].item()
            components = " | ".join([f"{component_scores[r.name][batch_idx].item():.3f}" for r in self.reward_fns])
            print(f"{smiles:<60} {total:<10.4f} {components}")
        
        total_rewards = torch.nan_to_num(total_rewards, nan=-0.1)

        return total_rewards, component_scores