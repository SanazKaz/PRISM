import torch
import pickle
from pathlib import Path
import tempfile
from typing import List, Set
import os
import warnings
import numpy as np

from rdkit import Chem
import prolif as plf
from Bio.PDB import PDBIO, PDBParser

from src.prism.reward.scorer import BaseReward
from src.prism.utils import center_pocket_on_ligand_com

# Suppress warnings
warnings.filterwarnings("ignore")

def prepare_protein_for_prolif(prot_mol, target_chain="A"):
    """
    Sanitizes protein. Adds Hs and forces the specific Chain ID found in the filename.
    """
    try:
        prot_h = Chem.AddHs(prot_mol, addCoords=True)
        
        for atom in prot_h.GetAtoms():
            info = atom.GetPDBResidueInfo()
            
            if info is None:
                info = Chem.AtomPDBResidueInfo()
                info.SetResidueName("UNL")
                info.SetResidueNumber(1)
                info.SetIsHeteroAtom(False)
                info.SetName(f"X{atom.GetIdx()}")
                
                neighbors = atom.GetNeighbors()
                if neighbors:
                    nb_info = neighbors[0].GetPDBResidueInfo()
                    if nb_info:
                        info.SetResidueName(nb_info.GetResidueName())
                        info.SetResidueNumber(nb_info.GetResidueNumber())
                        info.SetIsHeteroAtom(nb_info.GetIsHeteroAtom())
                        info.SetName(f"H{atom.GetIdx()}")

            info.SetChainId(target_chain)
            atom.SetMonomerInfo(info)

        return prot_h
    except Exception as e:
        # print(f"Error preparing protein: {e}")
        return None
    
def calculate_prolif_fingerprint(pocket, ligand, fp_generator, base_name) -> Set[str]:
    """
    Calculates fingerprint and returns a SET of interaction strings.
    Example output: {'TYR109.A::Hydrophobic', 'ASP123.A::HBAcceptor'}
    """
    # Extract Chain ID
    try:
        parts = base_name.split('_')
        chain_id = parts[2] if len(parts) >= 3 else "A"
    except:
        chain_id = "A"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_prot:
        temp_prot_path = tmp_prot.name
    
    try:
        io = PDBIO()
        io.set_structure(pocket)
        io.save(temp_prot_path)
        
        prot_rdkit = Chem.MolFromPDBFile(temp_prot_path, removeHs=False, flavor=1)
        if prot_rdkit is None:
            prot_rdkit = Chem.MolFromPDBFile(temp_prot_path, removeHs=False)
        if prot_rdkit is None: return set()

        protein_clean = prepare_protein_for_prolif(prot_rdkit, target_chain=chain_id)
        if protein_clean is None: return set()
            
        protein_mol = plf.Molecule(protein_clean)
        
        if ligand.GetNumConformers() == 0: return set()
        ligand_with_h = Chem.AddHs(ligand, addCoords=True)
        lig_mol = plf.Molecule(ligand_with_h)
        
        # 1. Run Calculation (DISABLE PROGRESS BAR HERE)
        fp_generator.run_from_iterable([lig_mol], protein_mol, progress=False)
        
        # 2. Extract Result as a Set of Strings
        df = fp_generator.to_dataframe()
        active_interactions = set()
        
        if not df.empty:
            # Iterate through columns (active interactions)
            # Column format usually: (ligand, protein_residue, interaction_type)
            for col in df.columns:
                # Check if interaction exists (Value is True)
                if df[col].values[0]: 
                    # Create a unique key: "Residue::Interaction"
                    # col[1] is the Protein Residue (e.g. TYR109.A)
                    # col[2] is the Interaction Type (e.g. Hydrophobic)
                    key = f"{col[1]}::{col[2]}"
                    active_interactions.add(key)
        
        return active_interactions
            
    except Exception as e:
        # print(f"Error in calculation: {e}")
        return set()
    finally:
        if os.path.exists(temp_prot_path):
            try: os.remove(temp_prot_path)
            except OSError: pass

class InteractionFingerprintsReward(BaseReward):
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.reference_fingerprints = []
        self.checks_done = 0  # <--- Counter for your 5 checks
        
        self.fp_generator = plf.Fingerprint([
            "HBDonor", "HBAcceptor", "PiStacking", 
            "Hydrophobic", "CationPi", "Anionic", "Cationic"
        ])
        
        data_root = Path(self.dataset_info['datadir']).parent
        pockets_dir = data_root / '02_preprocessed' / 'pocket_files'
        sdf_dir = data_root / '02_preprocessed' / 'sdf_files'
        
        print(f"[ProLIFp Init] Loading from {sdf_dir}")
        
        for sdf_path in sdf_dir.glob('*.sdf'):
            base_name = sdf_path.stem
            pocket_path = pockets_dir / f"{base_name}_pocket.pdb"
            
            if not pocket_path.exists(): continue
            
            try:
                # 1. Calculate CENTERED Fingerprint (The Standard Way)
                pocket_centered, ligand_centered = center_pocket_on_ligand_com(
                    str(pocket_path), str(sdf_path)
                )
                
                if pocket_centered is None: continue
                
                fp_centered = calculate_prolif_fingerprint(
                    pocket_centered, 
                    ligand_centered, 
                    self.fp_generator,
                    base_name
                )
                
                # --- SANITY CHECK (Run 5 times) ---
                if fp_centered and self.checks_done < 15:
                    self._run_sanity_check(pocket_path, sdf_path, fp_centered, base_name)
                    self.checks_done += 1
                # ----------------------------------

                if fp_centered:
                    self.reference_fingerprints.append({
                        'name': base_name,
                        'fingerprint': fp_centered
                    })
            except Exception as e:
                pass 

        print(f"[ProLIFp Init] Loaded {len(self.reference_fingerprints)} references")

    def _run_sanity_check(self, raw_pocket_path, raw_sdf_path, fp_centered, base_name):
        """
        Loads the RAW (non-centered) files and compares their fingerprint
        to the centered version. They should match 100%.
        """
        try:
            # Load Raw Pocket
            parser = PDBParser(QUIET=True)
            raw_pocket = parser.get_structure("raw", str(raw_pocket_path))
            
            # Load Raw Ligand
            suppl = Chem.SDMolSupplier(str(raw_sdf_path), removeHs=False)
            raw_ligand = suppl[0]
            
            # Calculate Raw FP
            fp_raw = calculate_prolif_fingerprint(
                raw_pocket, 
                raw_ligand, 
                self.fp_generator, 
                base_name
            )
            
            # Compare Sets
            intersection = len(fp_centered.intersection(fp_raw))
            union = len(fp_centered.union(fp_raw))
            score = intersection / union if union > 0 else 0.0
            
            print(f"\n[Sanity Check] {base_name}")
            print(f"  Centered Interactions: {len(fp_centered)}")
            print(f"  Raw (Orig) Interactions: {len(fp_raw)}")
            print(f"  Consistency Score: {score:.3f} {'✅' if score==1.0 else '❌'}")
            
            if score < 1.0:
                print(f"  MISMATCH! Differences: {fp_centered.symmetric_difference(fp_raw)}")
                
        except Exception as e:
            print(f"  [Sanity Check] Failed: {e}")

    @property
    def name(self) -> str:
        return "interaction_fingerprints"

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        names = kwargs.get('names', [])
        
        # Fast lookup
        ref_lookup = {item['name']: item['fingerprint'] for item in self.reference_fingerprints}
        
        for idx, mol_gen in enumerate(molecules):
            try:
                if mol_gen is None:
                    scores.append(0.0)
                    continue

                sample_name = names[idx] if names else ""
                base_name = sample_name.split('_pocket')[0] if '_pocket' in sample_name else sample_name
                
                if base_name not in ref_lookup:
                    scores.append(0.0)
                    continue
                
                ref_set = ref_lookup[base_name]

                # Paths
                data_root = Path(self.dataset_info['datadir']).parent
                pocket_path = data_root / '02_preprocessed' / 'pocket_files' / f"{base_name}_pocket.pdb"
                sdf_path = data_root / '02_preprocessed' / 'sdf_files' / f"{base_name}.sdf"
                
                if not pocket_path.exists():
                    scores.append(0.0)
                    continue

                pocket_centered, _ = center_pocket_on_ligand_com(str(pocket_path), str(sdf_path))
                
                # Calculate Generated Set
                gen_set = calculate_prolif_fingerprint(
                    pocket_centered, 
                    mol_gen, 
                    self.fp_generator, 
                    base_name
                )
                
                # Tanimoto
                intersection = len(gen_set.intersection(ref_set))
                union = len(gen_set.union(ref_set))
                tanimoto = float(intersection) / float(union) if union > 0 else 0.0
                
                # Print Result
                print(f"[ProLIF] Mol {idx} ({base_name}): Tanimoto={tanimoto:.3f} | Ref={len(ref_set)} Gen={len(gen_set)}")
                scores.append(tanimoto) 
                    
            except Exception as e:
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32)