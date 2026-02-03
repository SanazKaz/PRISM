# src/prism/ppo_tuner/lightning_module.py

import pytorch_lightning as pl
import torch
from pathlib import Path
from torch.utils.data import DataLoader



# Import our new algorithm and the original model definition
from src.prism.ppo_tuner.ppo_algorithm import PPOAlgorithm
from src.models.diffsbdd.lightning_modules import LigandPocketDDPM
from src.prism.data_modules.lightning_datamodule import LigandPocketDataModule # You'll need to create this later
from src.prism.reward.factory import get_reward_manager
from src.prism.utils import build_molecules_from_batch
from val_analysis.smina_docking import SminaDocking
from val_analysis.metrics import MoleculeMetrics



class PPOFineTuner(pl.LightningModule):
    """
    This is the main LightningModule for PPO fine-tuning.
    It acts as a lightweight wrapper around the core PPOAlgorithm.
    """
    def __init__(self, config, node_histogram=None, warm_start_checkpoint=None, checkpoint_dir=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.automatic_optimization = False # Crucial for PPO
        
        
        device = torch.device("cuda" if self.config.gpus > 0 else "cpu")

        # Filter out PPO-specific and Lightning-specific parameters
        ddpm_config = {k: v for k, v in vars(self.config).items() 
                    if k not in ['ppo_params', 'enable_progress_bar', 
                                'num_sanity_val_steps', 'wandb_params',
                                'gpus', 'n_epochs', 'logdir', 'fp16', 
                                'run_identifier','reward_params'
                                ]}

        self.ddpm_model = LigandPocketDDPM(
            outdir=Path(self.config.logdir),
            node_histogram=node_histogram,
            **ddpm_config
        )
        self.ddpm_model.to(device)

        
        # Load pretrained weights if provided
        if warm_start_checkpoint is not None:
            # print(f"Loading pretrained DDPM weights from: {warm_start_checkpoint}")
            checkpoint = torch.load(warm_start_checkpoint, map_location='cpu', weights_only=False)
            # Extract state dict - handle both direct state_dict and lightning checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.ddpm_model.load_state_dict(state_dict, strict=False)
            # print("Successfully loaded pretrained weights!")
        
        self.dataset_info = self.ddpm_model.dataset_info.copy()
        self.dataset_info['datadir'] = self.config.datadir
    
        
        # 3. Instantiate the RewardManager
        self.reward_manager = get_reward_manager(
            config=self.config,
            dataset_info=self.dataset_info,
            ddpm_module=self.ddpm_model
        )

        # 2. Instantiate our self-contained PPOAlgorithm
        self.ppo_algorithm = PPOAlgorithm(
            policy_network=self.ddpm_model,
            reward_function=self.reward_manager,
            config=self.config,
            dataset_info=self.dataset_info,
            checkpoint_dir=checkpoint_dir
        )
        self.freeze_parameters()

    def freeze_parameters(self):
        print("[Init] Applying EGNN freezing strategy...")
        frozen_count = 0
        unfrozen_count = 0
        
        # Get trainable blocks from config, with a default fallback
        trainable_layers = self.config.ppo_params.freeze_except
        print(f"[Init] Trainable blocks: {trainable_layers}")
        
        for name, param in self.ppo_algorithm.policy_network.named_parameters():
            if any(x in name for x in trainable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1
        
        # Recount for verification
        for param in self.ppo_algorithm.policy_network.parameters():
            if param.requires_grad:
                unfrozen_count += 1
        
        print(f"[Init] {frozen_count} params frozen, {unfrozen_count} trainable.")
        
        
        
    def configure_optimizers(self):
        """
        Define the optimizer here so Lightning can track it.
        """
        # We access the internal DDPM parameters just like you did in the algorithm
        # Ensure we only optimize parameters that require grad (the freezing logic)
        params_to_optimize = filter(lambda p: p.requires_grad, self.ddpm_model.ddpm.parameters())
        
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.config.ppo_params.lr,
            eps=1e-8,
            weight_decay=1.0e-12,
            betas=(0.9, 0.999)
        )
        return optimizer
            

    def training_step(self, batch, batch_idx):
        """
        The training step is now incredibly clean. It just calls the algorithm.
        """
        opt = self.optimizers()
        # The PL Trainer handles the outer loop by calling this method repeatedly.
        # Our PPOAlgorithm handles the inner loops inside its train_step method.
        print(f"[DEBUG] Global Step: {self.global_step} | Current Epoch: {self.current_epoch}")
        
        logs = self.ppo_algorithm.train_step(
            pocket_batch=batch,
            current_epoch=self.current_epoch,
            optimizer=opt
        )
        # Log the metrics returned by the algorithm
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        
        return logs


    # --- Delegate other essential methods to the original model ---

    def validation_step(self, batch, batch_idx):
        # For validation, we use the original model's logic
        return self.ddpm_model.validation_step(batch, batch_idx)
    
    def _run_validation(self):
        """
        Runs validation on the validation set of pockets (same as train)
        Calculates metrics in analysis/metrics.py
        """
        
        self.ddpm_model.eval()
        
        try:
            with torch.no_grad():
            
                val_dataloader = self.trainer.datamodule.val_dataloader()
                
                n_eval_samples = self.config.eval_params.n_eval_samples  # 20 from config
                eval_batch_size = self.config.eval_params.eval_batch_size  # 20 from config
                
                val_batch = next(iter(val_dataloader))
                actual_batch_size = min(eval_batch_size, len(val_batch['num_pocket_nodes']))  # <--- FIX: min not len
                
                # val_batch_subset = {
                #     key: value[:actual_batch_size] if torch.is_tensor(value) 
                #     else value[:actual_batch_size] if isinstance(value, list)
                #     else value
                #     for key, value in val_batch.items()
                # }
                
                print(f"[Validation] Generating molecules for {actual_batch_size} pockets...")
                # We pass the validation batch through the collector
                get_ligand_and_pocket_fn = self.ddpm_model.get_ligand_and_pocket
                
                rollout_data = self.ppo_algorithm.collector.collect(
                    pocket_batch=val_batch,
                    current_epoch=self.current_epoch,
                    get_ligand_and_pocket_fn=get_ligand_and_pocket_fn
                )
                
                # 3. Extract molecules from rollout data
                if rollout_data['rewards'].numel() == 0:
                    print("[Validation] No valid molecules generated, skipping metrics.")
                    return
                
                xh_lig = rollout_data['molecules'][0]  # ligand features
                global_lig_mask = rollout_data['masks'][0]  # which atoms belong to which molecule
                
                print(f"[Validation] Generated {len(torch.unique(global_lig_mask))} molecules")
                            
                molecules, mol_to_batch_idx = build_molecules_from_batch(
                    xh_lig,
                    global_lig_mask,
                    self.dataset_info,
                    self.ddpm_model
                )
                
                print(f"[Validation] Successfully built {len(molecules)} valid RDKit molecules")
                
                # 5. Calculate metrics using analysis module
                names = val_batch.get('names', None)
                
                metrics_calculator = MoleculeMetrics(dataset_info=self.dataset_info)
                basic_metrics = metrics_calculator.evaluate_batch(molecules)
                
                docking_calculator = SminaDocking(dataset_info=self.dataset_info, local_opt=False, timeout=60)
                docking_metrics = docking_calculator.dock_batch(molecules, names=names)
                
                
                metrics = {**basic_metrics, **docking_metrics}
                                
                # 6. Log metrics to WandB
                val_metrics = {f"val/{key}": value for key, value in metrics.items()}
                self.log_dict(val_metrics, on_step=False, on_epoch=True)
                
                # Print summary
                print(f"\n[Validation Results - Epoch {self.current_epoch + 1}]")
                print("-" * 60)
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                print("-" * 60 + "\n")
                
        except Exception as e:
            print(f"[Validation] Error during validation: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always return to train mode
            self.ddpm_model.train()    
            
    
    def on_train_epoch_end(self):
        """
        called by lightning trainer after each training epoch.
        Triggers validation at specific intervals.
        """
        if (self.current_epoch + 1) % self.config.eval_epochs == 0:
            print(f"\n{'='*80}")
            print(f"Running validation at epoch {self.current_epoch + 1}")
            print(f"{'='*80}\n")
            self._run_validation() # custom method to handle validation
    
