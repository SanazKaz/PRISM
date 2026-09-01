"""
Molecular Metrics for Validation

Calculates molecular properties and PoseBusters validation for tracking during training.
Includes both fast drug-likeness metrics and structural validation checks.
"""

import torch
import numpy as np
from typing import List, Dict
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from posebusters import PoseBusters

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'models' / 'diffsbdd'))
from analysis.SA_Score.sascorer import calculateScore

RDLogger.DisableLog('rdApp.*')


class MoleculeMetrics:
    
    def __init__(self, dataset_info=None):
        self.dataset_info = dataset_info
        self.posebusters = PoseBusters(config="mol_fast")
    
    def calculate_qed(self, mol: Chem.Mol) -> float:
        try:
            custom_weights = (0.66, 0.46, 0.10, 0.61, 0.06, 0.40, 0.6, 0.95)
            return float(QED.qed(mol, w=custom_weights))
        except:
            return 0.0
    
    def calculate_sa_score(self, mol: Chem.Mol) -> float:
        try:
            sa_raw = calculateScore(mol)
            normalized = (10 - sa_raw) / 9
            return max(0.0, min(1.0, normalized))
        except:
            return 0.0
    
    def calculate_lipinski_violations(self, mol: Chem.Mol) -> int:
        try:
            rule_1 = Descriptors.ExactMolWt(mol) < 500
            rule_2 = Lipinski.NumHDonors(mol) <= 5
            rule_3 = Lipinski.NumHAcceptors(mol) <= 10
            logp = Crippen.MolLogP(mol)
            rule_4 = (-2 <= logp <= 5)
            rule_5 = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
            
            violations = sum([not rule for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]])
            return violations
        except:
            return 5
    
    def calculate_lipinski_pass(self, mol: Chem.Mol) -> int:
        try:
            violations = self.calculate_lipinski_violations(mol)
            return 5 - violations
        except:
            return 0
    
    def get_ring_info(self, mol: Chem.Mol) -> Dict[str, int]:
        try:
            ring_info = mol.GetRingInfo()
            n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
            n_total = rdMolDescriptors.CalcNumRings(mol)
            
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
            largest_ring = max(ring_sizes) if ring_sizes else 0
            
            return {
                'n_aromatic_rings': n_aromatic,
                'n_non_aromatic_rings': n_total - n_aromatic,
                'n_total_rings': n_total,
                'largest_ring_size': largest_ring
            }
        except:
            return {
                'n_aromatic_rings': 0,
                'n_non_aromatic_rings': 0,
                'n_total_rings': 0,
                'largest_ring_size': 0
            }
    
    @staticmethod
    def calculate_tanimoto_similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
        try:
            fp1 = Chem.RDKFingerprint(mol_a)
            fp2 = Chem.RDKFingerprint(mol_b)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def calculate_diversity(self, molecules: List[Chem.Mol]) -> float:
        if len(molecules) < 2:
            return 0.0
        
        try:
            div = 0
            total = 0
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    similarity = self.calculate_tanimoto_similarity(
                        molecules[i], molecules[j]
                    )
                    div += 1 - similarity
                    total += 1
            return div / total if total > 0 else 0.0
        except:
            return 0.0
    
    def run_posebusters(self, molecules: List[Chem.Mol]) -> Dict[str, float]:
        if not molecules:
            return {'posebusters_pass_rate': 0.0}
        
        try:
            results_df = self.posebusters.bust(molecules)
            
            check_columns = [col for col in results_df.columns 
                           if results_df[col].dtype == 'bool']
            
            if not check_columns:
                return {'posebusters_pass_rate': 0.0}
            
            results_df['passed_all_checks'] = results_df[check_columns].all(axis=1)
            pass_rate = results_df['passed_all_checks'].mean()
            
            return {'posebusters_pass_rate': float(pass_rate)}
            
        except Exception as e:
            print(f"[Metrics] PoseBusters error: {e}")
            return {'posebusters_pass_rate': 0.0}
    
    def calculate_molecule_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        if mol is None:
            return None
        
        try:
            props = {
                'qed': self.calculate_qed(mol),
                'sa_score': self.calculate_sa_score(mol),
                'lipinski_violations': self.calculate_lipinski_violations(mol),
                'lipinski_rules_passed': self.calculate_lipinski_pass(mol),
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'n_heavy_atoms': mol.GetNumHeavyAtoms(),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'logp': Crippen.MolLogP(mol),
            }
            
            ring_info = self.get_ring_info(mol)
            props.update(ring_info)
            
            return props
            
        except Exception as e:
            print(f"[Metrics] Error calculating properties: {e}")
            return None
    
    def evaluate_batch(self, molecules: List[Chem.Mol]) -> Dict[str, float]:
        if not molecules:
            return {
                'validity': 0.0,
                'n_molecules': 0,
                'posebusters_pass_rate': 0.0
            }
        
        all_props = []
        for mol in molecules:
            props = self.calculate_molecule_properties(mol)
            if props is not None:
                all_props.append(props)
        
        n_valid = len(all_props)
        n_total = len(molecules)
        
        metrics = {
            'validity': n_valid / n_total if n_total > 0 else 0.0,
            'n_molecules': n_total,
            'n_valid_molecules': n_valid
        }
        
        if n_valid == 0:
            metrics['posebusters_pass_rate'] = 0.0
            return metrics
        
        for key in all_props[0].keys():
            values = [p[key] for p in all_props]
            metrics[f'{key}_mean'] = np.mean(values)
        
        print("[Metrics] Calculating diversity...")
        diversity = self.calculate_diversity(molecules)
        metrics['diversity'] = diversity
        
        print("[Metrics] Running PoseBusters validation...")
        pb_metrics = self.run_posebusters(molecules)
        metrics.update(pb_metrics)
        
        return metrics
