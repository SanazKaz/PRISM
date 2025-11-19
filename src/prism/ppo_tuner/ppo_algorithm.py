# src/prism/ppo_tuner/ppo_algorithm.py

import torch
from torch.optim import Adam

# Import the clean components we just built
from .rollout_collector import RolloutCollector
from .rollout_buffer import RolloutBuffer
from .loss import compute_ppo_loss
from src.models.diffsbdd.lightning_modules import LigandPocketDDPM 
from utils import permute_timesteps
from tests.ppo_debug_utils import assert_same_ids, assert_latent_alignment, reset_seen_mb_ids

class PPOAlgorithm:
    """
    The main PPO algorithm class. It orchestrates the collector, buffer, and loss
    calculation to perform the PPO update. This is a framework-agnostic class.
    """
    def __init__(self, 
                 policy_network: LigandPocketDDPM, 
                 reward_function,
                 config, 
                 dataset_info, 
                 run_root):
        self.policy_network = policy_network
        self.config = config
        self.device = next(policy_network.parameters()).device
        self.reward_function = reward_function

        # Create the optimizer here, as it's tied to the algorithm
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.policy_network.ddpm.parameters()),
            lr=self.config.ppo_params.lr,
            eps=1e-8,
            weight_decay=1.0e-12,
            betas=(0.9, 0.999)
        )

        # NOTE: compose our clean components
        self.collector = RolloutCollector(
            policy_network=self.policy_network.ddpm,
            reward_function=self.reward_function, # We'll need to refactor the reward function next
            config=config
        )
        self.buffer = RolloutBuffer(config=config)
    
    def train_step(self, pocket_batch, current_epoch):
        """
        Performs one full step of the PPO outer loop.
        """
        # --- 1. Collect Experience ---
        get_ligand_and_pocket_fn = self.policy_network.get_ligand_and_pocket
        rollout_data = self.collector.collect(pocket_batch, current_epoch, get_ligand_and_pocket_fn)

        # --- 2. Store Experience and Compute Advantages ---
        self.buffer.load_rollout_data(rollout_data)
        if not self.buffer.data_loaded:
            print("Skipping training step due to no valid rollouts.")
            return {"train/policy_loss": 0}

        self.buffer.compute_advantages()
        
        # --- 2.5. GET THE ACTUAL ROLLOUT DATA BACK FROM BUFFER ---
        # We need to modify the buffer data directly, so get references
        rollout_data_for_permute = {
            'timesteps': self.buffer.timesteps,
            'old_log_probs': self.buffer.old_log_probs,
            'latents': self.buffer.latents,
            'next_latents': self.buffer.next_latents,
            'masks': self.buffer.masks,
            'advantages': self.buffer.advantages,
            'rewards': self.buffer.rewards,
            'raw_score': self.buffer.raw_scores,
            'molecules': self.buffer.molecules,
            'pocket_indices': self.buffer.pocket_indices
        }
        
        # --- 2.6. DETERMINE CURRENT K (from original code) ---
        current_k = self.config.ppo_params.num_train_timesteps   # Default: 64 or whatever is in config
        
        
        # --- 2.7. SLICE TO LAST K TIMESTEPS (from original: rollout_data[key] = rollout_data[key][:, -k:]) ---
        for key in ("latents", "next_latents", "old_log_probs", "timesteps"):
            if key in rollout_data_for_permute and rollout_data_for_permute[key] is not None:
                rollout_data_for_permute[key] = rollout_data_for_permute[key][:, -current_k:]
        
        # --- 2.8. PERMUTE TIMESTEPS (from original) ---
        with torch.no_grad():
            rollout_data_for_permute = permute_timesteps(rollout_data_for_permute, self.device)
            
        
        ######################### DEBUGGING #########################
        lig_mask, pocket_mask = rollout_data_for_permute['masks']
        assert_same_ids("training_step/after_permute", lig_mask, pocket_mask)
        assert_latent_alignment(
        "training_step/post_permute",
        rollout_data_for_permute['latents'],
        rollout_data_for_permute['next_latents'],
        lig_mask,
        pocket_mask,
        require_temporal=False  # permutation breaks temporal order
    )
        ######################### DEBUGGING #########################
        
        # --- 2.9. UPDATE BUFFER WITH PERMUTED DATA ---
        self.buffer.latents = rollout_data_for_permute['latents']
        self.buffer.next_latents = rollout_data_for_permute['next_latents']
        self.buffer.old_log_probs = rollout_data_for_permute['old_log_probs']
        self.buffer.timesteps = rollout_data_for_permute['timesteps']
        
        # --- 3. Run PPO Inner Epochs ---
        self.policy_network.ddpm.train()

        # Initialize trackers for logs
        total_loss, total_kl, total_clipfrac, total_entropy = 0, 0, 0, 0
        update_count = 0

        num_inner_epochs = self.config.ppo_params.num_inner_epochs
        for inner_epoch in range(num_inner_epochs):
            reset_seen_mb_ids() # reset the seen molecule ids for each inner epoch
            
            print(f"Outer epoch: {current_epoch}, Inner epoch: {inner_epoch}")
            
            step_every = self.config.ppo_params.gradient_accumulation_steps
            scale = step_every  # matching original code
            
            accumulation_count = 0
            epoch_total_loss = 0.0
            epoch_total_approx_kl = 0.0
            epoch_total_clipfrac = 0.0
            epoch_total_entropy = 0.0
            epoch_accumulation_steps = 0
            
            for minibatch in self.buffer.get_minibatches():
                # NOW USE current_k INSTEAD OF num_train_timesteps!
                for t_idx in range(current_k):  # <-- THIS IS KEY
                    policy_loss, approx_kl, clipfrac, entropy = compute_ppo_loss(
                        policy_network=self.policy_network.ddpm,
                        minibatch=minibatch,
                        timestep_idx=t_idx,
                        config=self.config
                    )
                    
                    scaled_loss = policy_loss / scale
                    scaled_loss.backward()
                    
                    accumulation_count += 1
                    
                    # Only perform optimization after accumulating for specified steps
                    if accumulation_count % step_every == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_network.ddpm.parameters(),
                            self.config.ppo_params.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    # Accumulate metrics
                    epoch_total_loss += policy_loss.detach().item()
                    epoch_total_approx_kl += approx_kl.item()
                    epoch_total_clipfrac += clipfrac.item()
                    epoch_total_entropy += entropy.item()
                    epoch_accumulation_steps += 1
            
            # After the loop, flush leftovers if any
            if accumulation_count % step_every != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.ddpm.parameters(),
                    self.config.ppo_params.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # Final logs for the entire outer step
        final_logs = {
            "train/total_loss_epoch": epoch_total_loss / max(epoch_accumulation_steps, 1),
            "train/approx_kl_epoch": epoch_total_approx_kl / max(epoch_accumulation_steps, 1),
            "train/clipfrac_epoch": epoch_total_clipfrac / max(epoch_accumulation_steps, 1),
            "train/entropy_epoch": epoch_total_entropy / max(epoch_accumulation_steps, 1),
            "train/reward_mean": self.buffer.rewards.mean().item(),
            "train/advantages_mean": self.buffer.advantages.mean().item(),
            "train/advantages_std": self.buffer.advantages.std().item(),
            "train/advantages_min": self.buffer.advantages.min().item(),
            "train/advantages_max": self.buffer.advantages.max().item(),
        }

        # --- NEW: Log individual reward components (QED, SuCOS, etc.) ---
        # We pull these directly from the rollout_data we collected earlier
        if 'component_scores' in rollout_data:
            for name, score_tensor in rollout_data['component_scores'].items():
                if score_tensor.numel() > 0:
                    # Log the mean of the raw scores
                    final_logs[f"train/reward_{name}_mean"] = score_tensor.mean().item()
        # ----------------------------------------------------------------

        print(f"total_loss epoch: {epoch_total_loss / max(epoch_accumulation_steps, 1)}")
        
        return final_logs
