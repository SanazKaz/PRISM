# src/prism/ppo_tuner/rollout_buffer.py

import torch
import torch.distributed as dist
import numpy as np
from tests.ppo_debug_utils import validate_minibatch, reset_seen_mb_ids
import hashlib




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
        """Normalise rewards into advantages.

        Two modes, selected by ``config.ppo.use_grpo_advantages`` (default
        False):
          - False: GLOBAL batch normalisation across all molecules in the
            rollout (pocket-agnostic). Robust to small batches and
            single-sample-per-pocket scenarios.
          - True:  GRPO — per-pocket (within-group) normalisation. Each
            pocket's samples are z-scored against only the other samples for
            that same pocket, removing pocket difficulty as a confound.
            Requires several samples per pocket to carry signal.

        The result is clipped to [-3, 3] to prevent outlier rewards from
        producing exploding gradient updates.
        """
        if not self.data_loaded:
            raise ValueError("Cannot compute advantages before loading data.")

        use_grpo = getattr(self.config.ppo, "use_grpo_advantages", False)
        if use_grpo:
            self.advantages = self._grpo_advantages()
        else:
            self.advantages = self._global_advantages()

        self.advantages = torch.clamp(self.advantages, min=-3.0, max=3.0)

    def _global_advantages(self):
        """Normalise rewards against global batch statistics (default).

        Global normalisation (across all molecules in the rollout, regardless
        of pocket) is robust to small batches and single-sample-per-pocket
        scenarios where per-pocket normalisation would produce NaNs.
        """
        if dist.is_initialized():
            # Compute global mean/std across all ranks so advantages are
            # comparable when DDP averages gradients.
            local_sum    = self.rewards.sum()
            local_sum_sq = (self.rewards ** 2).sum()
            local_count  = torch.tensor(float(self.rewards.numel()),
                                        device=self.rewards.device)
            dist.all_reduce(local_sum,    op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count,  op=dist.ReduceOp.SUM)
            global_mean = local_sum / local_count
            global_var  = (local_sum_sq / local_count) - global_mean ** 2
            global_std  = (global_var.clamp(min=0.0) + 1e-8).sqrt()
            batch_mean, batch_std = global_mean, global_std
        else:
            batch_mean = self.rewards.mean()
            batch_std  = self.rewards.std()

        if self.rewards.numel() > 1 and batch_std > 1e-6:
            return (self.rewards - batch_mean) / (batch_std + 1e-8)
        return self.rewards - batch_mean

    def _grpo_advantages(self):
        """GRPO: per-pocket (within-group) reward normalisation.

        All samples for a given pocket live on the same rank (DDP splits
        pockets across ranks), so group statistics are LOCAL — no cross-rank
        reduction is needed (unlike the global path). Groups with fewer than 2
        samples or zero variance are mean-subtracted (=> 0), contributing no
        gradient. This is the correct GRPO behaviour for a degenerate group
        and avoids the divide-by-zero NaN that naive per-pocket normalisation
        produced previously.
        """
        rewards    = self.rewards
        pocket_ids = self.pocket_indices
        advantages = torch.zeros_like(rewards)
        for pid in torch.unique(pocket_ids):
            mask  = pocket_ids == pid
            group = rewards[mask]
            group_mean = group.mean()
            if group.numel() > 1:
                group_std = group.std()
                if group_std > 1e-6:
                    advantages[mask] = (group - group_mean) / (group_std + 1e-8)
                else:
                    advantages[mask] = group - group_mean   # all-equal -> 0
            else:
                advantages[mask] = group - group_mean        # singleton -> 0
        return advantages


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
        ppo_batch_size = self.config.ppo.batch_size
        
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
