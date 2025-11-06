# src/prism/ppo_tuner/rollout_buffer.py

import torch
from torch_scatter import scatter_mean
import math
import numpy as np
from tests.ppo_debug_utils import validate_minibatch, reset_seen_mb_ids
import hashlib  # local import keeps top-of-file changes minimal




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
        A generator that yields minibatches matching the original training_step logic:
        - does NOT shuffle the molecule ID ordering here (positional slicing is preserved)
        - builds atom masks from selected_ids
        - slices per-molecule arrays positionally (start_idx:end_idx)
        - includes a byte-level latents checksum 'lat_hash' for each molecule in the minibatch
        """

        if not self.data_loaded or self.advantages is None:
            raise ValueError("Must load data and compute advantages before creating minibatches.")
            
        num_molecules = self.rewards.shape[0]
        ppo_batch_size = self.config.ppo_params.ppo_batch_size
        
        xh_lig_full, xh_pocket_full = self.molecules
        lig_mask_full, pocket_mask_full = self.masks

        # Get all unique molecule IDs from the mask — DO NOT shuffle here (matches old code)
        all_mol_ids_np = np.unique(lig_mask_full.cpu().numpy())
        print(f"All molecule IDs: {all_mol_ids_np}")

        # Create minibatches by positional slices over the unique IDs
        for i in range(0, num_molecules, ppo_batch_size):
            start_idx = i
            print(f"Start index: {start_idx}")
            end_idx = min(i + ppo_batch_size, num_molecules)
            print(f"End index: {end_idx}")
            # Pick molecule IDs for this minibatch (positional selection)
            selected_ids = all_mol_ids_np[start_idx:end_idx]
            
            # Create atom-based masks for minibatch
            lig_minibatch_mask = torch.zeros_like(lig_mask_full, dtype=torch.bool)
            pocket_minibatch_mask = torch.zeros_like(pocket_mask_full, dtype=torch.bool)
            for mol_id in selected_ids:
                lig_minibatch_mask |= (lig_mask_full == mol_id)
                pocket_minibatch_mask |= (pocket_mask_full == mol_id)
            
            # Per-atom slices
            mb_xh_lig = xh_lig_full[lig_minibatch_mask]
            mb_xh_pocket = xh_pocket_full[pocket_minibatch_mask]
            mb_latents = self.latents[lig_minibatch_mask]
            mb_next_latents = self.next_latents[lig_minibatch_mask]
            mb_lig_mask = lig_mask_full[lig_minibatch_mask]
            
            # Per-molecule (positional) slices
            mb_advantages = self.advantages[start_idx:end_idx]
            mb_old_log_probs = self.old_log_probs[start_idx:end_idx]
            mb_timesteps = self.timesteps[start_idx:end_idx]
            mb_rewards = self.rewards[start_idx:end_idx]
            mb_raw_score = self.raw_scores[start_idx:end_idx]
            mb_pocket_indices = self.pocket_indices[start_idx:end_idx]

            # Build minibatch dict mirroring original structure
            minibatch = {
                # Per-atom tensors: SLICED using boolean masks
                'molecules': (mb_xh_lig, mb_xh_pocket),
                'masks': (mb_lig_mask, pocket_mask_full[pocket_minibatch_mask]),
                'latents': mb_latents,
                'next_latents': mb_next_latents,

                # Per-molecule tensors: positional slices
                'advantages': mb_advantages,
                'old_log_probs': mb_old_log_probs,
                'timesteps': mb_timesteps,
                'rewards': mb_rewards,
                'raw_score': mb_raw_score,
                'pocket_indices': mb_pocket_indices,
            }

            # Compute byte-level checksum for each molecule in the minibatch (lat_hash)
            # This matches the old training-step checksum used for final integrity check.
            lat_hash_list = []
            for mol_id in selected_ids:
                local_atom_mask = (mb_lig_mask == mol_id)
                if not local_atom_mask.any():
                    # keep same semantics as old code: skip if no atoms for this mol_id
                    lat_hash_list.append(0)
                    continue
                # compute md5 of the latents bytes for this molecule and reduce to 32-bit int
                arr_bytes = mb_latents[local_atom_mask].detach().cpu().numpy().tobytes()
                hex8 = hashlib.md5(arr_bytes).hexdigest()[:8]
                h = int(hex8, 16) & 0x7FFFFFFF
                lat_hash_list.append(h)
            minibatch['lat_hash'] = torch.tensor(lat_hash_list, dtype=torch.long, device=mb_latents.device)

            # Validation 
            validate_minibatch(minibatch, tag=f"epoch{self.config.current_epoch if hasattr(self.config, 'current_epoch') else 0}_mb{i}")

            yield minibatch
