
import warnings
from typing import Callable
from pathlib import Path
import tempfile, os, shutil
import subprocess, sys
from collections import defaultdict
import os


import torch
from torch_scatter import scatter_mean
import numpy as np
from scipy.optimize import linear_sum_assignment
import utils
import math
from utils import batch_to_list
import time
import re
from typing import Optional, Dict, List
from Bio.PDB import PDBParser, PDBIO
from openbabel import pybel
from analysis.docking import smina_score

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, AllChem, DataStructs, rdMolDescriptors, Crippen, ChemicalFeatures
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig


from medchem.structural.lilly_demerits import LillyDemeritsFilters
from scoring.posebusters.posebusters import PoseBusters
from scoring.transformations import reverse_sigmoid, double_sigmoid, sigmoid
from utils import find_gt_files, save_generation_triplet, biopdb_structure_to_rdkit 
from scoring.SuCOS_master import calc_SuCOS_normalized as calc_SuCOS
from analysis.molecule_builder import build_molecule, process_molecule, center_pocket_on_ligand_com
from analysis.metrics import MoleculeProperties
from analysis.SA_Score.sascorer import calculateScore



class MedChemReward:
    """
    Class to compute rewards for generated molecular structures based on
    medicinal chemistry properties and 3D structures.
    Weights for different properties can be adjusted to balance
    the importance of each property in the final reward score.
    """
    def __init__(self,
                 w_qed: float = 1.0,
                 w_penalised_logp: float = 0.0,
                 w_ring_structure_bonus: float = 0.0,
                 w_pharmacophore_similarity: float = 0.0,
                 w_posebusters: float = 0.0,
                 w_sucos: float = 0.0,
                 w_sas_score: float = 0.0,
                 w_oxygen: float = 0.0,
                 w_medchem: float = 0.0,
                 demerit_threshold: int = 160,
                 w_smina: float = 0.0,
                 dataset_info=None,
                 sucos_score_mode: str = "best",
                 save_sdf: bool = True,
                 save_every_n: int = 1,
                 ddpm_module=None,
                 run_root: Path | None = None,
                 ecfp_center: float = 0.20,
                 prev_mean: float | None = None,
                 ema_alpha: float = 0.10
                 ):
        self.w_qed   = w_qed
        self.w_penalised_logp = w_penalised_logp
        self.w_ring_structure_bonus = w_ring_structure_bonus
        self.w_pharmacophore_similarity = w_pharmacophore_similarity
        self.w_sas_score = w_sas_score
        self.w_sucos = w_sucos
        self.w_oxygen = w_oxygen
        self.w_medchem = w_medchem
        self.w_posebusters = w_posebusters
        self.w_smina = w_smina
        self.sucos_score_mode = sucos_score_mode
        self.demerit_threshold = demerit_threshold
        self.dataset_info = dataset_info
        self.save_sdf = bool(save_sdf)
        self.save_every_n = max(1, int(save_every_n)) 
        self.ddpm_module = ddpm_module
        self.run_root = run_root
        self.ecfp_center = ecfp_center
        self.prev_mean = prev_mean # none on first batch, then EMA of composite reward
        self.ema_alpha = ema_alpha
        
        # metric registry → (fn, weight)
        self.metric_fns: dict[str, tuple[Callable, float]] = {
            "qed":   (self.qed_score,   self.w_qed),
            # "sucos": (self.sucos_score, self.w_sucos),
            # "medchem": (self.lilly_medchem_score, self.w_medchem),
            # "smina": (self.smina_docking, self.w_smina),
        }

    
    def build_molecules_from_batch(self, xh_lig, lig_mask, sdf_dir=None):
        """
        Build RDKit molecule objects from batched ligand tensors.

        Parameters:
        -----------
        xh_lig : torch.Tensor
            Tensor of ligand 3D coordinates and one-hot atom types. Shape: [N, 3 + A]
        lig_mask : torch.Tensor
            Tensor mapping each atom to a molecule index. Shape: [N]
        Returns:
        --------
        Tuple[List[Chem.Mol], Dict[int, int]]
            - List of successfully built and processed RDKit molecules
            - Mapping from molecule index in list to original batch index
        """
        if hasattr(self, 'ddpm_module') and hasattr(self.ddpm_module, 'virtual_nodes') and self.ddpm_module.virtual_nodes:
            atom_types = xh_lig[:, 3:].argmax(1)  # Get atom type indices
            vnode_mask = (atom_types == self.ddpm_module.virtual_atom)
            
            # Filter out virtual atoms
            xh_lig = xh_lig[~vnode_mask]
            lig_mask = lig_mask[~vnode_mask]
            
            # If all atoms were virtual (shouldn't happen, but safety check)
            if xh_lig.shape[0] == 0:
                return [], {}
        
        
        
        x = xh_lig[:, :3].detach().cpu()
        atom_type = torch.argmax(xh_lig[:, 3:], dim=1).detach().cpu()
        lig_mask = lig_mask.cpu()

        molecules = []
        molecule_to_batch_idx = {}

        for batch_idx, mol_pc in enumerate(zip(batch_to_list(x, lig_mask),
                                            batch_to_list(atom_type, lig_mask))):
            try:
                mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
                mol = process_molecule(
                    mol,
                    add_hydrogens=False,
                    sanitize=True,
                    relax_iter=0,
                    largest_frag=True
                )
                if mol is not None:
                    molecules.append(mol)
                    molecule_to_batch_idx[len(molecules)-1] = batch_idx
            except Exception as e:
                print(f"Failed to build molecule for batch index {batch_idx}: {str(e)}")
                continue

        return molecules, molecule_to_batch_idx
    
    def posebusters_score(self, mol: Chem.Mol) -> float:
        """
        Runs PoseBusters on a single molecule and returns normalized score.
        """
        if mol is None:
            return 0.0
        bust = PoseBusters(config="gen")
        result = bust(mol)
        

    def lilly_medchem_score(self, mol: Chem.Mol) -> float:
        """
        Runs Lilly MedChem Rules on a single molecule and returns normalized score.
        
        Parameters:
        -----------
        mol : rdkit.Chem.Mol
            RDKit molecule object with or without explicit hydrogens
            
        Returns:
        --------
        float
            Normalized score where 1.0 is best and 0.0 is worst
        """
        if mol is None:
            return 0.0
        
        try:
            # Remove explicit hydrogens for proper scoring
            mol_no_h = Chem.RemoveHs(mol)
            
            # # Check for 3-membered rings
            # ring_info = mol_no_h.GetRingInfo().AtomRings()
            # for ring in ring_info:
            #     if len(ring) == 3:  # Check size of each individual ring
            #         print("Rejected: Contains 3-membered ring")
            #         return 0.0
            
            # Create the filter object (can be reused) 
            dfilters = LillyDemeritsFilters(
            **{"dthresh": self.demerit_threshold,
               "min_atoms": 15,
               "hard_max_atoms": 50,
               "max_size_rings": 6,
               "min_num_rings": 1,
               "max_num_rings": 4,
               "max_size_chain": 6,
               }
        )
            
            # Run the filter
            result = dfilters(mols=[mol_no_h])
            
            # Extract the first result
            result_row = result.iloc[0]
            
            # Get the demerit score, status, and reasons
            demerit_score = result_row['demerit_score']
            status = result_row['status']
            reasons = result_row['reasons']
            
            soft_threshold = self.demerit_threshold / 2.0  # e.g., 80 for a 160 threshold
            
            if  np.isnan(demerit_score):
                demerit_score = self.demerit_threshold * 2  # assign penalty score
                

            score = reverse_sigmoid(demerit_score, k=0.1, center=soft_threshold)
            
            
            # # Optional: print diagnostic information
            print(f"Status: {status}, Demerits: {demerit_score}, Reasons: {reasons}")
            print(f"Final score: {score:.3f}")

            return score
            
        except Exception as e:
            print(f"Error in Lilly MedChem scoring: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def qed_score(self, mol: Chem.Mol, mw: float | None = None) -> float:
        """
        QED reward - square root scaling for better early training
        doesnt work even with 10 heavy atoms threshold - is this because
        the graph is 3D? this is a 2D reward after all.
        """
        try:
            # custom weights for QED score to reduce rigidity weight
            custom_weights = (0.66, 0.46, 0.05, 0.61, 0.06, 0.30, 0.48, 0.95)
            q = float(qed(mol, w=custom_weights))
            # sigmoid_q = sigmoid(q, k=10.0, center=0.5)
            # print(f"[DEBUG] sigmoid_q: {sigmoid_q}")
            print(f"[DEBUG] raw qed: {q}")
            return q
        except Exception as e:
            print(f"Error in QED scoring: {e}")
            return 0.0
        
    
    def sucos_score(self, ref_file, prb_file) -> float:
        
        try:
            # Calculate SuCOS score
            sucos_score = calc_SuCOS.main(ref_file, prb_file)
            

            return sucos_score
        
        except Exception as e:
            print(f"Failed to calculate SuCOS score: {str(e)}")
            return 0.0
    
        
    def pharmacophore_similarity_hungarian(self,
                                gen_mol: Chem.Mol,
                                ref_mol_centered: Chem.Mol,
                                sigma: float = 1.5,
                                max_dist: float = 4.0,
                                keep_families: set[str] = {'Donor', 'Acceptor', 'Aromatic'}
                                ) -> float:
        """
        Pharmacophore similarity with one-to-one Hungarian matching.
        Combines recall (cover reference sites), precision (avoid spamming),
        and quality (how close the matched pairs are).
        Returns score in [0,1].
        """

        # --- extract features ---
        fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        ff = ChemicalFeatures.BuildFeatureFactory(fdef_path)
        family_map = {'Donor': 0, 'Acceptor': 1, 'Aromatic': 2}

        def get_feats(mol):
            coords, labels = [], []
            for feat in ff.GetFeaturesForMol(mol):
                fam = feat.GetFamily()
                if fam in keep_families:
                    pos = feat.GetPos(confId=feat.GetActiveConformer())
                    coords.append([float(pos.x), float(pos.y), float(pos.z)])
                    labels.append(family_map[fam])
            if not coords:
                return np.empty((0, 3)), np.empty((0,))
            return np.array(coords), np.array(labels)

        G, Gl = get_feats(gen_mol)
        R, Rl = get_feats(ref_mol_centered)

        if len(R) == 0 or len(G) == 0:
            return 0.0

        # --- pairwise distances and label mask ---
        D = np.linalg.norm(R[:, None, :] - G[None, :, :], axis=2)
        same = (Rl[:, None] == Gl[None, :])

        big = 1e6
        def soft_kernel(d): return np.exp(-(d * d) / (sigma * sigma))
        C = np.where(same & (D <= max_dist), 1.0 - soft_kernel(D), big)

        # --- Hungarian assignment ---
        n = max(len(R), len(G))
        C_pad = np.full((n, n), big, float)
        C_pad[:len(R), :len(G)] = C
        r_idx, g_idx = linear_sum_assignment(C_pad)

        mask = (r_idx < len(R)) & (g_idx < len(G)) & (C_pad[r_idx, g_idx] < big * 0.5)
        r_idx, g_idx = r_idx[mask], g_idx[mask]
        m = len(r_idx)
        
        if m == 0:
            # optional debug: see why no match happened
            if len(R) > 0 and len(G) > 0:
                if np.random.rand() < 0.02:  # ~2% of the time
                    same_mask = (Rl[:, None] == Gl[None, :])
                    if same_mask.any():
                        min_dist_same = np.min(D[same_mask])
                    else:
                        min_dist_same = "NA"
                    print(f"[PH4 DEBUG] no match: |R|={len(R)} |G|={len(G)} min_dist_same={min_dist_same}")
            return 0.0
        
        # --- precision, recall, quality ---
        recall = m / len(R)
        precision = m / len(G)
        quality = float(soft_kernel(D[r_idx, g_idx]).mean())

        # --- final blended score ---
        alpha, beta, gamma = 0.4, 0.4, 0.2
        score = alpha * recall + beta * precision + gamma * quality

        return float(score)


    def smina_docking(self, rdmols, receptor_file,  lig_sdf_path, local_opt = False):
        score = smina_score(rdmols, receptor_file, lig_sdf_path, local_opt = local_opt)
        if score is None:
            print(f"[WARN] No smina score calculated for {rdmols} against {receptor_file}")
            return 0.0  # Invalid molecule marker
        
        raw_score = score[0]
        print(f"[DEBUG] smina score: {raw_score}")
        
        # GDPO-style normalization: [-20, 0] -> [0, 1]
        clipped = max(-15, min(0, raw_score))
        normalized = -clipped / 15.0
        
        return normalized
    
     
    def ring_structure_bonus(
        self,
        mol: Chem.Mol,
        max_rings: int = 4,
        max_ring_size: int = 6,
        bonus: float = 1.0,
        ) -> float:
        """
        Give a +bonus if the molecule passes simple ring constraints:
        - number of rings <= max_rings
        - largest ring size <= max_ring_size
        Otherwise return 0.0.

        Returns:
            float: 0.0 or `bonus`
        """
        if mol is None:
            return 0.0

        try:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()  # tuple of rings (each ring = tuple(atom indices))
            num_rings = len(atom_rings)
            largest = max((len(r) for r in atom_rings), default=0)

            if num_rings > max_rings or largest > max_ring_size:
                return 0.0
            return float(bonus)
        except Exception:
            return 0.0
    
    
    def penalized_logp(self, molecule):
        
        """Calculates the penalized logP of a molecule.

        Refactored from
        https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
        See Junction Tree Variational Autoencoder for Molecular Graph Generation
        https://arxiv.org/pdf/1802.04364.pdf
        Section 3.2
        Penalized logP is defined as:
        y(m) = logP(m) - SA(m) - cycle(m)
        y(m) is the penalized logP,
        logP(m) is the logP of a molecule,
        SA(m) is the synthetic accessibility score,
        cycle(m) is the largest ring size minus by six in the molecule.

        Args:
            molecule: Chem.Mol. A molecule.

        Returns:
            Float. The penalized logP value.

        """
        log_p = Descriptors.MolLogP(molecule)
        sas_score = sascorer.calculateScore(molecule)
        largest_ring_size = get_largest_ring_size(molecule)
        cycle_score = max(largest_ring_size - 6, 0)
        return log_p - sas_score - cycle_score
    
    def sas_score(self, molecule):
        sas_score = calculateScore(molecule)
        normalized = (10 - sas_score) / 9
        print(f"[DEBUG] raw_sas: {sas_score:.2f}, normalized: {normalized:.2f}")

        
        return normalized
    
    
    
    def composite_reward(
        self,
        xh_lig:      torch.Tensor,
        xh_pocket:   torch.Tensor,
        lig_mask:    torch.Tensor,
        pocket_mask: torch.Tensor,
        ref_mols:    Optional[Dict[int, Chem.Mol]] = None,
        current_epoch: Optional[int] = None,
        names: Optional[List[str]] = None
        ) -> torch.Tensor:
        """
        Compute QED + SuCOS + Lilly MedChem reward for each generated ligand.
        GT pocket & ligand are automatically centred to COM = 0.

        • files are saved through generation_saver.save_generation_triplet(…)
        • reference ligand & pocket written **once** per tag
        • generated ligand written every `self.save_every_n` occurrences
        """
        # ------------------------------------------------------------------
        # 0) setup
        # ------------------------------------------------------------------
        if names is None:
            raise ValueError("`names` must be provided")

        if ref_mols is None:
            ref_mols = {}

        device         = xh_lig.device
        unique_indices = torch.unique(lig_mask)              # one per mol
        batch_size     = len(unique_indices)
        rewards        = torch.zeros(batch_size, device=device)
        raw_score      = torch.zeros(batch_size, device=device)

        # ------------------------------------------------------------------
        # 1) build RDKit molecules from generated batch & centre them
        # ------------------------------------------------------------------
        molecules, mol_to_batch_idx = self.build_molecules_from_batch(
            xh_lig, lig_mask
        )
        rewards = torch.full((batch_size,), -0.1, device=device)
        if not molecules:
            print("[MedChemReward] No valid molecules built — returning default reward -0.1")
            return rewards, rewards

        temp_dir = tempfile.mkdtemp() # remove

        try:
            success_count = 0

            for local_idx, gen_mol in enumerate(molecules):
                try:
                    if local_idx >= len(mol_to_batch_idx):
                        print(f"[WARN] skip ligand {local_idx}: no valid RDKit mol ⇒ no batch-idx")
                        continue     
                    
                    batch_idx   = mol_to_batch_idx[local_idx]
                    global_id   = int(unique_indices[batch_idx].item())
                                        
                    # pocket names list is per-pocket, not per-ligand
                    if global_id < len(names):
                        sample_name = str(names[global_id])
                    else:
                        sample_name = str(names[0])          # reuse the first (only) pocket name
                    
                    lig_sdf_path, poc_pdb_path = find_gt_files(sample_name)
                    
                    # center the pocket on the ligand
                    pocket_structure_centered, ref_mol_centered = center_pocket_on_ligand_com(poc_pdb_path, lig_sdf_path)
                    
                    sup = Chem.SDMolSupplier(str(lig_sdf_path), removeHs=False)
                    
                    ref_mol = next((m for m in sup if m is not None), None) # original uncentered ligand
                    
                    #remove if not needed
                    sample_core = re.split(r"_pocket10\.pdb", sample_name, 1)[0]   # drop the tail
                    sample_core = sample_core.replace(".sdf", "")                  # drop trailing .sdf
                    safe_tag    = sample_core.replace("/", "_")
                    
                                        
                    # 2. If it PASSES, calculate the main objective reward (Tanimoto).
                                        
                    # QED_score = self.qed_score(gen_mol) 
                    # Lilly_score = self.lilly_medchem_score(gen_mol) 
                    # reward_composite = Lilly_score * self.w_medchem + cyclopropane_score
                    # fcfp4_score, fcfp4_tanimoto = self.fcfp4_similarity(gen_mol, ref_mol, current_epoch)
                    # reward_composite = fcfp4_score * self.w_fcfp4_similarity
                    
                    # pharmacophore_score = self.pharmacophore_satisfaction_score(gen_mol, ref_mol_centered, th=2.0)
                    qed = self.qed_score(gen_mol)
                    
                    reward_composite = qed * self.w_qed 
                                                            
                    rewards[batch_idx] = reward_composite
                    raw_score[batch_idx] = reward_composite
                    
                    num_atoms = gen_mol.GetNumAtoms()
                    print(f"reward_composite: {reward_composite}, num_atoms: {num_atoms}, smiles: {Chem.MolToSmiles(gen_mol)}")
                    
                    tag_for_saving = f"{safe_tag}_lig{local_idx}"
                    
                    if current_epoch == 0:
                        print(f"tag_for_saving: {tag_for_saving}")
                        print(f"refmol: {Chem.MolToSmiles(ref_mol_centered)}")

                    # ---------------- 6) optional saving ------------------
                    if self.save_sdf and (success_count % self.save_every_n == 0):
                        save_generation_triplet(
                            run_root=self.run_root,              # <- set in __init__
                            epoch=current_epoch,
                            tag=tag_for_saving,
                            gen_mol=gen_mol,
                            ref_mol=ref_mol_centered,
                            pocket_structure=pocket_structure_centered,
                            pocket_xyz=None,
                            save_ref_and_pocket_once=True
                        )

                    success_count += 1

                except Exception as e:
                    print(f"[MedChemReward] ERROR mol {local_idx}: {e}")

            # --- NEW REWARD SHAPING LOGIC --- remove later 
            # This happens *after* all raw scores for the batch have been calculated
            
            # 1. Create a mask to identify valid molecules (scores are not the -0.1 penalty)
            # valid_mask = raw_score > -0.1
            
            # # 2. If there are any valid molecules, calculate their mean score
            # if valid_mask.any():
            #     valid_raw_scores = raw_score[valid_mask]
            #     mean_reward_for_batch = valid_raw_scores.mean()
                
                # 3. Center the rewards: subtract the batch mean from all valid scores.
                #    This creates positive rewards for above-average mols and negative for below-average.
                #    The result is stored in the `rewards` tensor.
                # rewards[valid_mask] = valid_raw_scores - mean_reward_for_batch
                
                # print(f"[DEBUG] Reward Shaping: Batch Mean={mean_reward_for_batch:.3f}. Rewards are now centered around 0.")
            # --- END OF NEW LOGIC ---
            
            # # --- ADD THIS NEW LOOP TO SEE SHAPED REWARDS --- added for sucos
            # print("--- Individual Shaped Rewards for Batch ---")
            # for local_idx, batch_idx in mol_to_batch_idx.items():
            #     if valid_mask[batch_idx]: # Only print for valid molecules
            #         mol = molecules[local_idx]
            #         raw_s = raw_score[batch_idx].item()
            #         shaped_r = rewards[batch_idx].item()
            #         print(f"[SHAPED REWARD] smiles: {Chem.MolToSmiles(mol)}, raw: {raw_s:.3f}, shaped: {shaped_r:+.3f}")
            # print("-----------------------------------------")
            # # --- END OF NEW LOOP ---

        
             # Your EMA calculation for logging (this is good and should stay)
            valid_rewards_for_log = raw_score[raw_score != -0.1]
            if valid_rewards_for_log.numel() > 0:
                batch_mean_for_logging = float(valid_rewards_for_log.mean().item())
            else:
                batch_mean_for_logging = 0.0 if self.prev_mean is None else self.prev_mean

            if self.prev_mean is None:
                self.prev_mean = batch_mean_for_logging
            else:
                a = self.ema_alpha
                self.prev_mean = a * batch_mean_for_logging + (1.0 - a) * self.prev_mean
            print(f"[DEBUG] EMA of composite reward: {self.prev_mean:.4f}, batch mean on valid mols: {batch_mean_for_logging:.4f}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return rewards, raw_score
    
    
# TODO: SAVE individual reward results to a file so i can plot them later
