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
from src.prism.models.diffsbdd_policy import DiffSBDDPolicy
from src.prism.models.targetdiff_factory import load_targetdiff_policy
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

        model_type = getattr(self.config, 'model_type', 'diffsbdd')

        if model_type == 'targetdiff':
            self.policy, self.dataset_info = self._build_targetdiff_policy(
                device, warm_start_checkpoint
            )
            self.ddpm_model = None  # no LigandPocketDDPM when using TargetDiff
        else:
            self.policy, self.ddpm_model, self.dataset_info = self._build_diffsbdd_policy(
                device, node_histogram, warm_start_checkpoint
            )

        self.dataset_info['datadir'] = self.config.datadir

        # Instantiate the RewardManager
        self.reward_manager = get_reward_manager(
            config=self.config,
            dataset_info=self.dataset_info,
            ddpm_module=self.policy,  # policy satisfies the virtual-node interface
        )

        # 2. Instantiate our self-contained PPOAlgorithm
        self.ppo_algorithm = PPOAlgorithm(
            policy_network=self.policy,
            reward_function=self.reward_manager,
            config=self.config,
            dataset_info=self.dataset_info,
            checkpoint_dir=checkpoint_dir,
        )
        self.freeze_parameters()

    # ------------------------------------------------------------------
    # Private model builders
    # ------------------------------------------------------------------

    def _build_diffsbdd_policy(self, device, node_histogram, warm_start_checkpoint):
        ddpm_config = {k: v for k, v in vars(self.config).items()
                       if k not in ['ppo_params', 'enable_progress_bar',
                                    'num_sanity_val_steps', 'wandb_params',
                                    'gpus', 'n_epochs', 'logdir', 'fp16',
                                    'run_identifier', 'reward_params',
                                    'docking_params', 'model_type']}
        ddpm_module = LigandPocketDDPM(
            outdir=Path(self.config.logdir),
            node_histogram=node_histogram,
            **ddpm_config,
        )
        ddpm_module.to(device)
        if warm_start_checkpoint is not None:
            ckpt = torch.load(warm_start_checkpoint, map_location='cpu', weights_only=False)
            ddpm_module.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        policy = DiffSBDDPolicy(ddpm_module)
        dataset_info = ddpm_module.dataset_info.copy()
        return policy, ddpm_module, dataset_info

    def _build_targetdiff_policy(self, device, warm_start_checkpoint):
        checkpoint_path = getattr(self.config, 'targetdiff_checkpoint', warm_start_checkpoint)
        if checkpoint_path is None:
            raise ValueError(
                "TargetDiff requires a checkpoint. Set config.targetdiff_checkpoint "
                "or pass warm_start_checkpoint to PPOFineTuner."
            )
        policy = load_targetdiff_policy(
            checkpoint_path=checkpoint_path,
            device=device,
            protein_atom_feature_dim=getattr(self.config, 'targetdiff_protein_feat_dim', 27),
            ligand_atom_feature_dim=getattr(self.config, 'targetdiff_ligand_atom_types', 13),
        )
        # TargetDiff uses CrossDocked atom type set (add_aromatic, 13 classes).
        # We reuse the DiffSBDD dataset_info structure for reward scoring;
        # the atom decoder below maps TargetDiff indices to element symbols.
        dataset_info = {
            'atom_decoder': ['H', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'N',
                              'O', 'O', 'O', 'F', 'P', 'P', 'P', 'P',
                              'S', 'S', 'S', 'S', 'S', 'Cl'],
            'atom_encoder': {},   # populated from decoder below
            'colors_dic': [],
            'radius_dic': [],
        }
        dataset_info['atom_encoder'] = {
            sym: i for i, sym in enumerate(dataset_info['atom_decoder'])
        }
        return policy, dataset_info

    # ------------------------------------------------------------------

    def freeze_parameters(self):
        print("[Init] Applying freezing strategy...")
        frozen_count = 0
        unfrozen_count = 0

        trainable_layers = self.config.ppo_params.freeze_except
        print(f"[Init] Trainable blocks: {trainable_layers}")

        for name, param in self.policy.named_parameters():
            if any(x in name for x in trainable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1

        for param in self.policy.parameters():
            if param.requires_grad:
                unfrozen_count += 1

        print(f"[Init] {frozen_count} params frozen, {unfrozen_count} trainable.")
        
        
        
    def configure_optimizers(self):
        """
        Define the optimizer here so Lightning can track it.
        """
        params_to_optimize = filter(lambda p: p.requires_grad, self.policy.parameters())
        
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
        if self.ddpm_model is not None:
            return self.ddpm_model.validation_step(batch, batch_idx)
        # TargetDiff: molecule generation + metrics handled in _run_validation
        return {}
    
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
                get_ligand_and_pocket_fn = self.policy.get_ligand_and_pocket
                
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
                    self.policy,
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
            if self.ddpm_model is None:
                print(f"[Validation] Skipping DiffSBDD validation — TargetDiff mode.")
                return
            print(f"\n{'='*80}")
            print(f"Running validation at epoch {self.current_epoch + 1}")
            print(f"{'='*80}\n")
            self._run_validation()
    
