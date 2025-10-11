# src/prism/ppo_tuner/ppo_algorithm.py

import torch
from torch.optim import Adam

# Import the clean components we just built
from .rollout_collector import RolloutCollector
from .rollout_buffer import RolloutBuffer
from .loss import compute_ppo_loss
from src.models.diffsbdd.lightning_modules import LigandPocketDDPM # Assuming this will be the model's new home
from src.prism.rewards.mol_properties import DummyMedChemReward

class PPOAlgorithm:
    """
    The main PPO algorithm class. It orchestrates the collector, buffer, and loss
    calculation to perform the PPO update. This is a framework-agnostic class.
    """
    def __init__(self, policy_network: LigandPocketDDPM, config, dataset_info, run_root):
        self.policy_network = policy_network
        self.config = config
        self.device = next(policy_network.parameters()).device
        self.reward_function = DummyMedChemReward(dataset_info=dataset_info, ddpm_module=policy_network.ddpm)

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
        This includes collecting rollouts and running multiple inner training epochs.
        """
        # --- 1. Collect Experience ---
        # NOTE: We pass the model's helper function for data processing
        get_ligand_and_pocket_fn = self.policy_network.get_ligand_and_pocket
        
        # TODO: The reward function needs to be passed to the collector.
        # For now, we assume the collector handles it.
        # This will be the next refactoring step.
        rollout_data = self.collector.collect(pocket_batch, current_epoch, get_ligand_and_pocket_fn)

        # --- 2. Store Experience and Compute Advantages ---
        self.buffer.load_rollout_data(rollout_data)
        if not self.buffer.data_loaded:
            print("Skipping training step due to no valid rollouts.")
            return {"train/policy_loss": 0} # Return a dummy log

        self.buffer.compute_advantages()
        
        # --- 3. Run PPO Inner Epochs ---
        self.policy_network.ddpm.train()

        # Initialize trackers for logs
        total_loss, total_kl, total_clipfrac, total_entropy = 0, 0, 0, 0
        update_count = 0

        num_inner_epochs = self.config.ppo_params.num_inner_epochs
        for inner_epoch in range(num_inner_epochs):
            for minibatch in self.buffer.get_minibatches():
                
                num_train_timesteps = self.config.ppo_params.num_train_timesteps
                for t_idx in range(num_train_timesteps):
                
                    policy_loss, approx_kl, clipfrac, entropy = compute_ppo_loss(
                        policy_network=self.policy_network.ddpm,
                        minibatch=minibatch,
                        timestep_idx=t_idx,
                        config=self.config
                    )
                    
                    # --- Backpropagation and Optimization ---
                    # This logic is moved from the old training_step
                    # The LightningModule will call `backward` on the final loss
                    # For now, let's just calculate the loss. The LightningModule will handle the rest.
                    # Or, if we want this class to do it all...
                    
                    scaled_loss = policy_loss / self.config.ppo_params.gradient_accumulation_steps
                    
                    # We'll need the LightningModule to call backward, or we do it here.
                    # For a truly standalone class, we do it here.
                    scaled_loss.backward()

                    update_count += 1
                    if update_count % self.config.ppo_params.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_network.ddpm.parameters(),
                            self.config.ppo_params.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Accumulate logs
                    total_loss += policy_loss.item()
                    total_kl += approx_kl.item()
                    total_clipfrac += clipfrac.item()
                    total_entropy += entropy.item()

        # Final logs for the entire outer step
        num_loss_computations = max(1, update_count)
        final_logs = {
            "train/policy_loss": total_loss / num_loss_computations,
            "train/approx_kl": total_kl / num_loss_computations,
            "train/clipfrac": total_clipfrac / num_loss_computations,
            "train/entropy": total_entropy / num_loss_computations,
            "train/reward_mean": self.buffer.rewards.mean().item(),
            "train/advantages_mean": self.buffer.advantages.mean().item(),
        }
        
        return final_logs