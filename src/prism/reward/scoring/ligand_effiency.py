import torch
import numpy as np
import math
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.prism.reward.scorer import BaseReward
from val_analysis.smina_docking import SminaDocking


class LigandEfficiencyReward(BaseReward):
    """
    SMINA Ligand Efficiency Reward
    
    LE = |docking_score| / number_of_heavy_atoms
    
    Based on An's nc3D-DQN implementation:
    - LE of 0.3 kcal/mol/HA is the gold standard for drug-like molecules
    - Linear reward scaling from 0.0 to 0.3
    
    Units: kcal/mol/heavy atom (HA)
    """
    
    # Gold standard LE for drug-like molecules
    TARGET_LE = 0.3  # kcal/mol/HA
    
    def __init__(self, smina_path: str = "/data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static",
                 local_opt: bool = False, timeout: int = 45, dataset_info: dict = None):
        """
        Args:
            smina_path: Path to SMINA executable
            local_opt: Use local optimization (recommended: True)
            timeout: Timeout per docking in seconds
        """
        self.smina_path = smina_path
        self.local_opt = local_opt
        self.timeout = timeout
        self.docker = None  # Will be initialized on first call
        self.dataset_info = dataset_info
        
    @property
    def name(self) -> str:
        return "ligand_efficiency"
    
    def _initialize_docker(self, dataset_info):
        """Initialize SminaDocking with dataset info."""
        if self.docker is None:
            self.docker = SminaDocking(
                smina_path=self.smina_path,
                dataset_info=dataset_info,
                local_opt=self.local_opt,
                timeout=self.timeout
            )

    def _calculate_le_reward(self, docking_score: float, heavy_atoms: int, current_epoch: int = 0) -> float:
        """
        Sigmoid-transformed docking reward that provides gradient signal
        even for molecules with steric clashes.
        
        Updated to handle two-stage rewards:
        1. For score > 0: Exponential decay to reward clash reduction.
        2. For score <= 0: Sharper sigmoid to reward binding affinity.
        """
        if heavy_atoms == 0 or heavy_atoms < 10:
            return 0.0
        
        # --- Curriculum logic ---
        # Early on, we are more "lenient" (higher temperature) to find the pocket.
        # As epochs progress, we sharpen the reward to demand better affinity.
        if current_epoch < 50:
            temperature = 3.0
            target_score = -5.0
        else:
            temperature = 2.0  # Sharper
            target_score = -8.0 # Stricter
            
        # --- Stage 1: The "Physics" Stage (Positive scores) ---
        if docking_score > 0:
            # Gentle exponential decay so +50 is worse than +5
            # We scale this so the max reward at score=0 is around 0.1 - 0.15
            clash_penalty = 0.15 * math.exp(-docking_score / 15.0)
            reward = clash_penalty
        
        # --- Stage 2: The "Chemistry" Stage (Negative scores) ---
        else:
            # Standard Sigmoid transformation
            reward = 1.0 / (1.0 + math.exp((docking_score - target_score) / temperature))
        
        print(f"[Docking] Score: {docking_score:.2f}, HA: {heavy_atoms}, "
              f"Epoch: {current_epoch}, Reward: {reward:.3f}")
        
        return float(reward)
    
    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Calculate ligand efficiency for each molecule.
        
        Requires kwargs:
            - dataset_info: Dict with 'datadir' key
            - names: List of pocket names for each molecule
            - current_epoch: Passed from RewardManager
        
        Returns:
            Tensor of rewards in [0, 1]
        """
        dataset_info = kwargs.get('dataset_info')
        names = kwargs.get('names', [])
        current_epoch = kwargs.get('current_epoch', 0)
        
        if dataset_info is None:
            print("[LigandEfficiency] WARNING: No dataset_info provided, returning zeros")
            return torch.zeros(len(molecules), dtype=torch.float32)
        
        self._initialize_docker(dataset_info)
        
        if self.docker.pocket_dir is None or self.docker.sdf_dir is None:
            print("[LigandEfficiency] WARNING: Docking directories not available")
            return torch.zeros(len(molecules), dtype=torch.float32)
        
        scores = []
        
        for i, mol in enumerate(molecules):
            if mol is None:
                scores.append(0.0)
                continue
            
            pocket_name = names[i] if i < len(names) else None
            if pocket_name is None:
                scores.append(0.0)
                continue
            
            try:
                base_name = self.docker._parse_pocket_name(pocket_name)
                pocket_path = self.docker.pocket_dir / f"{base_name}_pocket.pdb"
                ref_ligand_path = self.docker.sdf_dir / f"{base_name}.sdf"
                
                if not pocket_path.exists() or not ref_ligand_path.exists():
                    scores.append(0.0)
                    continue
                
                docking_extract = self.docker.dock_molecule(
                    mol=mol,
                    receptor_pdb_path=str(pocket_path),
                    ref_ligand_path=str(ref_ligand_path)
                )
                                                
                docking_score = docking_extract.get('score')
                
                # REMOVED: if docking_score >= 0.0: return 0.0
                # We now allow the _calculate_le_reward function to handle all scores.
                if docking_score is None:
                    scores.append(0.0)
                    continue
                
                heavy_atoms = mol.GetNumHeavyAtoms()
                fq_reward = self._calculate_le_reward(docking_score, heavy_atoms, current_epoch)
                
                scores.append(float(fq_reward))
                
            except Exception as e:
                print(f"[LigandEfficiency] Error docking molecule {i}: {e}")
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)