# src/prism/ppo_tuner/ppo_algorithm.py

import torch
import torch.distributed as dist
from torch.optim import Adam
import csv
import os
from pathlib import Path

# Import the clean components we just built
from .rollout_collector import RolloutCollector
from .rollout_buffer import RolloutBuffer
from .loss import compute_ppo_loss
from src.prism.models.base_policy import BaseDiffusionPolicy
from utils import permute_timesteps
from tests.ppo_debug_utils import assert_same_ids, assert_latent_alignment, reset_seen_mb_ids

class PPOAlgorithm:
    """
    The main PPO algorithm class. It orchestrates the collector, buffer, and loss
    calculation to perform the PPO update. This is a framework-agnostic class.
    """
    def __init__(self,
                 policy_network: BaseDiffusionPolicy,
                 reward_function,
                 config,
                 dataset_info,
                 checkpoint_dir,
                 ref_policy=None):
        self.policy_network = policy_network
        self.ref_policy = ref_policy
        self.config = config
        self.device = next(policy_network.parameters()).device
        self.reward_function = reward_function
        self.checkpoint_dir = checkpoint_dir
        
        self.log_dir = Path(checkpoint_dir) / "training_logs"
        self.csv_path = self.log_dir / "training_metrics.csv"
        _rank = dist.get_rank() if dist.is_initialized() else 0
        if _rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_csv()


        self.collector = RolloutCollector(
            policy_network=self.policy_network,
            reward_function=self.reward_function,
            config=config,
            ref_policy=self.ref_policy,
        )
        self.buffer = RolloutBuffer(config=config)
        
    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        # Define all possible metric names
        self.metric_headers = [
            'train/total_loss_epoch',
            'train/approx_kl_epoch',
            'train/clipfrac_epoch',
            'train/entropy_epoch',
            'train/kl_penalty_epoch',
            'train/reward_mean',
            'train/reward_top10_mean',
            'train/reward_std',
            'train/advantages_mean',
            'train/advantages_std',
            'train/advantages_min',
            'train/advantages_max',
        ]
        
        # Write header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch'] + self.metric_headers)
                writer.writeheader()
    
    def _log_to_csv(self, epoch, metrics):
        _rank = dist.get_rank() if dist.is_initialized() else 0
        if _rank != 0:
            return
        row = {'epoch': epoch}
        row.update(metrics)
        
        # Get dynamic headers (e.g., reward components), maintaining order
        dynamic_headers = [k for k in metrics.keys() if k not in self.metric_headers]
        
        # Use consistent fieldnames: epoch + base metrics + dynamic metrics
        all_fieldnames = ['epoch'] + self.metric_headers + dynamic_headers
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writerow(row)
    
    def train_step(self, pocket_batch, current_epoch, optimizer, backward_fn=None):
        """
        Performs one full step of the PPO outer loop.
        """
        if backward_fn is None:
            backward_fn = lambda loss: loss.backward()
        # Refresh device — Lightning may have moved the model after __init__ (DDP wrapping).
        self.device = next(self.policy_network.parameters()).device
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
        current_k = self.config.ppo.train_timesteps
        
        
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
        self.policy_network.train()

        # Initialize trackers for logs
        total_loss, total_kl, total_clipfrac, total_entropy, total_kl_penalty = 0, 0, 0, 0, 0
        update_count = 0

        num_inner_epochs = self.config.ppo.num_inner_epochs
        target_kl = getattr(self.config.ppo, 'target_kl', None)
        kl_early_stop = False

        for inner_epoch in range(num_inner_epochs):
            if kl_early_stop:
                break

            reset_seen_mb_ids()

            step_every = self.config.ppo.gradient_accumulation_steps
            scale = step_every  # matching original code

            accumulation_count = 0
            epoch_total_loss = 0.0
            epoch_total_approx_kl = 0.0
            epoch_total_clipfrac = 0.0
            epoch_total_entropy = 0.0
            epoch_total_kl_penalty = 0.0
            epoch_accumulation_steps = 0

            for minibatch in self.buffer.get_minibatches():
                for t_idx in range(current_k):
                    policy_loss, approx_kl, clipfrac, entropy = compute_ppo_loss(
                        policy_network=self.policy_network,
                        minibatch=minibatch,
                        timestep_idx=t_idx,
                        config=self.config,
                    )

                    # Sync KL across ranks so every rank makes the same early-stop decision.
                    if dist.is_initialized():
                        dist.all_reduce(approx_kl, op=dist.ReduceOp.AVG)

                    scaled_loss = policy_loss / scale
                    backward_fn(scaled_loss)

                    accumulation_count += 1

                    if accumulation_count % step_every == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_network.parameters(),
                            self.config.ppo.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_total_loss += policy_loss.detach().item()
                    epoch_total_approx_kl += approx_kl.item()
                    epoch_total_clipfrac += clipfrac.item()
                    epoch_total_entropy += entropy.item()
                    kl_coef = getattr(self.config.ppo, 'kl_coef', 0.0)
                    epoch_total_kl_penalty += (kl_coef * approx_kl.item())
                    epoch_accumulation_steps += 1

                    if target_kl is not None and approx_kl.item() > target_kl:
                        kl_early_stop = True
                        break

                if kl_early_stop:
                    break

            # Flush any remaining accumulated gradients.
            if accumulation_count % step_every != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(),
                    self.config.ppo.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # All-gather rewards and advantages so every rank computes metrics from
        # the full global sample, not just its local shard.  With sync_dist=True
        # in log_dict, PL will mean-reduce across ranks — since every rank now
        # holds the identical global value that reduction is a no-op, giving
        # correct statistics even for non-linear aggregates (max, std, top-k).
        if dist.is_initialized():
            _world = dist.get_world_size()
            _g_rew = [torch.zeros_like(self.buffer.rewards) for _ in range(_world)]
            _g_adv = [torch.zeros_like(self.buffer.advantages) for _ in range(_world)]
            dist.all_gather(_g_rew, self.buffer.rewards)
            dist.all_gather(_g_adv, self.buffer.advantages)
            global_rewards = torch.cat(_g_rew)
            global_advantages = torch.cat(_g_adv)
        else:
            global_rewards = self.buffer.rewards
            global_advantages = self.buffer.advantages

        # Top-10 mean: mean of the best min(10, N) rewards in the global rollout.
        _sorted, _ = torch.sort(global_rewards, descending=True)
        _top_k = min(10, len(_sorted))
        reward_top10_mean = _sorted[:_top_k].mean().item()

        # Validity: invalid molecules are left at exactly -0.1 (the penalty set in
        # RewardManager when reconstruction fails). Valid molecules start at 0.0
        # before reward accumulation, so > -0.1 is a reliable validity mask.
        n_attempted = global_rewards.numel()
        n_valid = (global_rewards > -0.1).sum().item()
        validity_rate = n_valid / max(n_attempted, 1)

        # Final logs for the entire outer step
        final_logs = {
            "train/n_attempted": n_attempted,
            "train/n_valid": n_valid,
            "train/validity_rate": validity_rate,
            "train/total_loss_epoch": epoch_total_loss / max(epoch_accumulation_steps, 1),
            "train/approx_kl_epoch": epoch_total_approx_kl / max(epoch_accumulation_steps, 1),
            "train/clipfrac_epoch": epoch_total_clipfrac / max(epoch_accumulation_steps, 1),
            "train/entropy_epoch": epoch_total_entropy / max(epoch_accumulation_steps, 1),
            "train/kl_penalty_epoch": epoch_total_kl_penalty / max(epoch_accumulation_steps, 1),
            "train/reward_mean": global_rewards.mean().item(),
            "train/reward_top10_mean": reward_top10_mean,
            "train/reward_std": global_rewards.std().item(),
            "train/advantages_mean": global_advantages.mean().item(),
            "train/advantages_std": global_advantages.std().item(),
            "train/advantages_min": global_advantages.min().item(),
            "train/advantages_max": global_advantages.max().item(),
        }

        # --- Log individual reward components (QED, SuCOS, etc.) ---
        if 'component_scores' in rollout_data:
            for name, score_tensor in rollout_data['component_scores'].items():
                if score_tensor.numel() > 0:
                    # Log the mean of the raw scores
                    final_logs[f"train/reward_{name}_mean"] = score_tensor.mean().item()
        # ----------------------------------------------------------------

        print(f"[epoch {current_epoch}] total_loss={epoch_total_loss / max(epoch_accumulation_steps, 1):.4f}  approx_kl={epoch_total_approx_kl / max(epoch_accumulation_steps, 1):.4f}")
        self._log_to_csv(current_epoch, final_logs)
        
        return final_logs
