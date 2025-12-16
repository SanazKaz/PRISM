"""
SMINA Docking Module

Handles molecular docking calculations using SMINA.
Separate from metrics calculation to maintain single responsibility.
"""

import os
import re
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from rdkit import Chem


class SminaDocking:
    """
    Handler for SMINA docking calculations.
    
    Responsible for:
    - Running SMINA executable
    - Managing temporary files
    - Parsing docking scores
    - Batch docking operations
    """
    
    def __init__(self, smina_path: str = "/data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static",
                 dataset_info: Dict = None, local_opt: bool = False, timeout: int = 60):
        """
        Initialize SMINA docking handler.
        
        Args:
            smina_path: Path to SMINA executable
            dataset_info: Dataset information containing datadir for finding pockets/ligands
        """
        self.smina_path = smina_path
        self.dataset_info = dataset_info
        self.local_opt = local_opt
        self.timeout = timeout
        if not os.path.exists(smina_path):
            print(f"[WARNING] SMINA executable not found at {smina_path}")
            print("[WARNING] Docking will be disabled")
        
        if dataset_info:
            self.data_root = Path(dataset_info['datadir']).parent
            self.pocket_dir = self.data_root / '02_preprocessed' / 'pocket_files'
            self.sdf_dir = self.data_root / '02_preprocessed' / 'sdf_files'
        else:
            self.pocket_dir = None
            self.sdf_dir = None
    
    @staticmethod
    def _extract_affinity(stdout: str) -> Optional[float]:
        """
        Parse SMINA stdout for affinity score.
        
        Args:
            stdout: SMINA output text
            
        Returns:
            Affinity in kcal/mol, or None if not found
        """
        m = re.search(r"Affinity:\s+([+-]?\d+(?:\.\d+)?)", stdout)
        if m:
            return float(m.group(1))
        
        m = re.search(r"^\s*1\s+([+-]?\d+(?:\.\d+)?)", stdout, re.MULTILINE)
        if m:
            return float(m.group(1))
        
        return None
    
    def dock_molecule(self, mol: Chem.Mol, receptor_pdb_path: str,
                     ref_ligand_path: str) -> Optional[float]:
        """
        Dock a single molecule to a receptor.
        
        Args:
            mol: RDKit molecule to dock
            receptor_pdb_path: Path to receptor PDB file
            ref_ligand_path: Path to reference ligand for autobox definition
            local_opt: Perform local optimization (slower but more accurate)
            timeout: Maximum time in seconds for docking
            
        Returns:
            Docking score in kcal/mol, or None if docking fails
        """
        if not os.path.exists(self.smina_path):
            return None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as lig_tmp:
                writer = Chem.SDWriter(lig_tmp.name)
                writer.write(mol)
                writer.close()
                
                if self.local_opt:
                    cmd = (
                        f"{self.smina_path} -l {lig_tmp.name} -r {receptor_pdb_path} "
                        f"--autobox_ligand {ref_ligand_path} "
                        f"--exhaustiveness 4 --num_modes 1"
                    )
                else:
                    cmd = f'{self.smina_path} -l {lig_tmp.name} -r {receptor_pdb_path} --score_only'
                
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.timeout
                )
                
                os.unlink(lig_tmp.name)
                
                if result.returncode != 0:
                    return None
                
                score = self._extract_affinity(result.stdout)
                return score
                
        except subprocess.TimeoutExpired:
            print(f"[Docking] Timeout during docking")
            return None
        except Exception as e:
            print(f"[Docking] Error: {e}")
            return None
    
    def _parse_pocket_name(self, name: str) -> str:
        """
        Extract base pocket name from complex naming scheme.
        
        Args:
            name: Complex pocket name (e.g., '7dfp_D_SIP_pocket_only.pdb_...')
            
        Returns:
            Base name (e.g., '7dfp_D_SIP')
        """
        if '_pocket_only.pdb' in name:
            return name.split('_pocket_only.pdb')[0]
        elif '_pocket' in name:
            return name.split('_pocket')[0]
        else:
            return name
    
    def dock_batch(self, molecules: List[Chem.Mol], names: List[str],
                   max_failures: int = 5) -> Dict[str, float]:
        """
        Dock a batch of molecules to their corresponding pockets.
        
        Args:
            molecules: List of RDKit molecules
            names: List of pocket names corresponding to each molecule
            max_failures: Stop early if this many consecutive failures occur
            
        Returns:
            Dictionary with docking statistics (mean, std, median, count)
        """
        if not molecules or not names:
            return self._empty_results()
        
        if self.pocket_dir is None or self.sdf_dir is None:
            print("[Docking] Dataset directories not configured")
            return self._empty_results()
        
        if not self.pocket_dir.exists() or not self.sdf_dir.exists():
            print("[Docking] Pocket or SDF directories not found")
            return self._empty_results()
        
        scores = []
        consecutive_failures = 0
        
        for mol, name in zip(molecules, names):
            if consecutive_failures >= max_failures:
                print(f"[Docking] Stopping after {max_failures} consecutive failures")
                break
            
            try:
                base_name = self._parse_pocket_name(name)
                
                pocket_path = self.pocket_dir / f"{base_name}_pocket.pdb"
                ref_ligand_path = self.sdf_dir / f"{base_name}.sdf"
                
                if not pocket_path.exists():
                    consecutive_failures += 1
                    continue
                
                if not ref_ligand_path.exists():
                    consecutive_failures += 1
                    continue
                
                score = self.dock_molecule(
                    mol,
                    str(pocket_path),
                    str(ref_ligand_path),
                )
                
                if score is not None:
                    scores.append(score)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    
            except Exception as e:
                consecutive_failures += 1
                continue
        
        if not scores:
            return self._empty_results()
        
        return {
            'smina_score_mean': float(np.mean(scores)),
            'smina_score_std': float(np.std(scores)),
            'smina_score_median': float(np.median(scores)),
            'smina_n_docked': len(scores),
            'smina_best_score': float(min(scores))
        }
    
    @staticmethod
    def _empty_results() -> Dict[str, float]:
        """Return empty results dict when docking is unavailable."""
        return {
            'smina_score_mean': 0.0,
            'smina_score_std': 0.0,
            'smina_score_median': 0.0,
            'smina_n_docked': 0,
            'smina_best_score': 0.0
        }