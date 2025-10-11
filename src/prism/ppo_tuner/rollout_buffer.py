# src/prism/ppo_tuner/rollout_buffer.py

import torch
from torch_scatter import scatter_mean
import math

class RolloutBuffer:
    """
    A buffer to store trajectories for PPO and compute advantages.
    It also provides an iterator for minibatches.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        """ Clears all stored data. """
        self.rewards = None
        self.raw_scores = None
        self.pocket_indices = None
        self.advantages = None
        self.molecules = None
        self.masks = None
        self.latents = None
        self.next_latents = None
        self.old_log_probs = None
        self.timesteps = None
        self.data_loaded = False

    def load_rollout_data(self, rollout_data: dict):
        """
        Loads a complete dictionary of rollout data from the RolloutCollector.
        """
        if not rollout_data or rollout_data['rewards'].numel() == 0:
            print("WARNING: RolloutBuffer received empty data. Skipping.")
            self.reset()
            return

        self.rewards = rollout_data['rewards']
        self.raw_scores = rollout_data['raw_score']
        self.pocket_indices = rollout_data['pocket_indices']
        self.molecules = rollout_data['molecules']
        self.masks = rollout_data['masks']
        self.latents = rollout_data['latents']
        self.next_latents = rollout_data['next_latents']
        self.old_log_probs = rollout_data['old_log_probs']
        self.timesteps = rollout_data['timesteps']
        self.data_loaded = True
        

    def compute_advantages(self):
        """
        Computes advantages on a per-pocket basis. This is the method
        from your original PPOTrainer, now living in its logical home.
        """
        if not self.data_loaded:
            raise ValueError("Cannot compute advantages before loading data.")

        # Determine the number of unique pockets on this rank.
        num_pockets = int(self.pocket_indices.max().item()) + 1

        # Calculate mean reward per pocket
        pocket_reward_mean = scatter_mean(self.rewards, self.pocket_indices, dim=0, dim_size=num_pockets)

        # Calculate standard deviation per pocket
        pocket_reward_mean_sq = scatter_mean(self.rewards.pow(2), self.pocket_indices, dim=0, dim_size=num_pockets)
        pocket_reward_var = pocket_reward_mean_sq - pocket_reward_mean.pow(2)
        pocket_reward_std = torch.sqrt(torch.clamp(pocket_reward_var, min=0)) + 1e-8

        # Expand per-pocket stats back to the full rewards tensor
        expanded_mean = pocket_reward_mean[self.pocket_indices]
        expanded_std = pocket_reward_std[self.pocket_indices]

        # Normalize rewards per-pocket to get advantages
        advantages = (self.rewards - expanded_mean) / expanded_std
        self.advantages = torch.clamp(advantages, min=-3, max=3)
        
        # Optional gating/top-k logic
        if self.config.ppo_params.top_k:
            self._apply_top_k_gating()

        print("Advantages computed and stored in buffer.")


    def _apply_top_k_gating(self):
        """ Zeros out advantages for samples that are not in the top-k by reward. """
        KEEP_FRAC = 0.30
        keep_mask = torch.zeros_like(self.advantages, dtype=torch.bool)
        unique_pockets = torch.unique(self.pocket_indices)

        for pid in unique_pockets:
            idx = (self.pocket_indices == pid).nonzero(as_tuple=True)[0]
            if idx.numel() == 0: continue
            
            k = max(1, int(math.ceil(KEEP_FRAC * idx.numel())))
            
            local_scores = self.rewards[idx]
            top_local_indices = torch.topk(local_scores, k=k, largest=True, sorted=False).indices
            global_keep_indices = idx[top_local_indices]
            keep_mask[global_keep_indices] = True

        self.advantages = self.advantages * keep_mask.float()
        print(f"Top-K advantage gating applied. Kept {keep_mask.sum()}/{keep_mask.numel()} samples.")


    def get_minibatches(self):
        """
        A generator that yields shuffled minibatches of the stored data.
        Matches the original training_step minibatching logic exactly.
        """
        if not self.data_loaded or self.advantages is None:
            raise ValueError("Must load data and compute advantages before creating minibatches.")
            
        num_molecules = self.rewards.shape[0]
        mol_indices = torch.randperm(num_molecules, device=self.rewards.device)
        
        ppo_batch_size = self.config.ppo_params.ppo_batch_size
        
        xh_lig_full, xh_pocket_full = self.molecules
        lig_mask_full, pocket_mask_full = self.masks
        
        for start_idx in range(0, num_molecules, ppo_batch_size):
            end_idx = start_idx + ppo_batch_size
            mb_mol_indices = mol_indices[start_idx:end_idx]  # e.g., [15, 0, 7, 27, 23]
            
            # Create boolean masks for atoms belonging to selected molecules
            # This matches: lig_minibatch_mask |= (lig_mask == mol_id)
            mb_lig_atom_mask = torch.isin(lig_mask_full, mb_mol_indices)
            mb_pocket_atom_mask = torch.isin(pocket_mask_full, mb_mol_indices)
            
            # SLICE all per-atom tensors to only include atoms from selected molecules
            minibatch = {
                # Per-atom tensors: SLICED
                'molecules': (xh_lig_full[mb_lig_atom_mask], xh_pocket_full[mb_pocket_atom_mask]),
                'masks': (lig_mask_full[mb_lig_atom_mask], pocket_mask_full[mb_pocket_atom_mask]),
                'latents': self.latents[mb_lig_atom_mask],
                'next_latents': self.next_latents[mb_lig_atom_mask],
                
                # Per-molecule tensors: SLICED by molecule indices
                'advantages': self.advantages[mb_mol_indices],
                'old_log_probs': self.old_log_probs[mb_mol_indices],
                'timesteps': self.timesteps[mb_mol_indices],
                'rewards': self.rewards[mb_mol_indices],
                'raw_score': self.raw_scores[mb_mol_indices],
                'pocket_indices': self.pocket_indices[mb_mol_indices],
            }
            
            yield minibatch