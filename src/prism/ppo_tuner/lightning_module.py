# src/prism/ppo_tuner/lightning_module.py

import pytorch_lightning as pl
import torch
from pathlib import Path
from torch.utils.data import DataLoader



# Import our new algorithm and the original model definition
from src.prism.ppo_tuner.ppo_algorithm import PPOAlgorithm
from src.models.diffsbdd.lightning_modules import LigandPocketDDPM
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule # You'll need to create this later

class PPOFineTuner(pl.LightningModule):
    """
    This is the main LightningModule for PPO fine-tuning.
    It acts as a lightweight wrapper around the core PPOAlgorithm.
    """
    def __init__(self, config, node_histogram, warm_start_checkpoint=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.automatic_optimization = False # Crucial for PPO
        
        device = torch.device("cuda" if self.config.gpus > 0 else "cpu")


        # Filter out PPO-specific and Lightning-specific parameters
        ddpm_config = {k: v for k, v in vars(self.config).items() 
                    if k not in ['ppo_params', 'enable_progress_bar', 
                                'num_sanity_val_steps', 'wandb_params', 'gpus', 'n_epochs', 'logdir', 'fp16']}

        self.ddpm_model = LigandPocketDDPM(
            outdir=Path(self.config.logdir, self.config.run_identifier),
            node_histogram=node_histogram,
            **ddpm_config
        )
        
        self.ddpm_model.to(device)

        
        # Load pretrained weights if provided
        if warm_start_checkpoint is not None:
            # print(f"Loading pretrained DDPM weights from: {warm_start_checkpoint}")
            checkpoint = torch.load(warm_start_checkpoint, map_location='cpu')
            # Extract state dict - handle both direct state_dict and lightning checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.ddpm_model.load_state_dict(state_dict, strict=False)
            # print("Successfully loaded pretrained weights!")

        # 2. Instantiate our self-contained PPOAlgorithm
        self.ppo_algorithm = PPOAlgorithm(
            policy_network=self.ddpm_model,
            config=self.config,
            dataset_info=self.ddpm_model.dataset_info,
            run_root=self.config.logdir
        )
        
    def on_train_start(self):
        """
        Freezes everything except the last 2 EGNN blocks (e_block_3 and e_block_4)
        """
        # print("[on_train_start] Applying EGNN freezing strategy...")
        
        frozen_count = 0
        unfrozen_count = 0
        
        for name, param in self.ppo_algorithm.policy_network.named_parameters():
            # Keep last 2 EGNN blocks trainable
            if any(x in name for x in ['e_block_3', 'e_block_4']):
                param.requires_grad = True
                # print(f"  KEEPING TRAINABLE: {name}")
            else:
                # Freeze everything else
                param.requires_grad = False
                frozen_count += 1
        
        # Count final trainable parameters
        for param in self.ppo_algorithm.policy_network.parameters():
            if param.requires_grad:
                unfrozen_count += 1
        
        trainable_percent = (unfrozen_count / (frozen_count + unfrozen_count)) * 100
        print(f"[TRAINING] {trainable_percent:.1f}% of parameters are trainable")
                

    def training_step(self, batch, batch_idx):
        """
        The training step is now incredibly clean. It just calls the algorithm.
        """
        # The PL Trainer handles the outer loop by calling this method repeatedly.
        # Our PPOAlgorithm handles the inner loops inside its train_step method.
        
        opt = self.optimizers()
        
        logs = self.ppo_algorithm.train_step(
            pocket_batch=batch,
            current_epoch=self.current_epoch
        )
        
        # Log the metrics returned by the algorithm
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False)
        
        return logs

    def configure_optimizers(self):
        # The optimizer is created and owned by the PPOAlgorithm
        return self.ppo_algorithm.optimizer

    # --- Delegate other essential methods to the original model ---

    def validation_step(self, batch, batch_idx):
        # For validation, we use the original model's logic
        return self.ddpm_model.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        return self.ddpm_model.on_validation_epoch_end()
    
