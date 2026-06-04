import os
import re
import tempfile
import numpy as np
import torch
from pathlib import Path
from ..utils import find_gt_files

import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import subprocess
from Bio.PDB import PDBIO, Structure



def calculate_smina_score(pdb_file, sdf_file, ref_ligand_file, local_opt=True):
   """
   Calculate SMINA docking scores for ligand-receptor interactions.
   """
   SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
   SMINA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "analysis", "smina.static")
   if local_opt:
       cmd = (
            f"{SMINA_PATH} -l {sdf_file} -r {pdb_file} "
            f"--autobox_ligand {ref_ligand_file} "
            f"--exhaustiveness 4 --num_modes 1"
        )
   else:
       cmd = f'{SMINA_PATH} -l {sdf_file} -r {pdb_file} --score_only'

   # --- DEBUGGING STATEMENTS ---
   print("\n--- Running SMINA ---")
   print(f"  Receptor: {pdb_file}")
   print(f"  Ligand: {sdf_file}")
   print(f"  Command: {cmd}")
   # --- END DEBUGGING ---

   result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
   
   # --- DEBUGGING STATEMENTS ---
   if result.returncode != 0:
       print(f"[ERROR] Smina failed with return code {result.returncode}")
       print(f"  Stderr: {result.stderr}")
   print(f"  Smina stdout:\n{result.stdout}")
   # --- END DEBUGGING ---
   
   score = extract_affinity(result.stdout)
   print(f"  Extracted Score: {score}")
   print("---------------------\n")

   if score is None:
    print("[WARN] no affinity found ")
    return [-1.0]
   
   return [score]


def smina_score(rdmols, receptors, ref_ligand_file=None, local_opt=False):
    """
    Calculate smina score. This function is now robust to receiving either
    a file path for the receptor or an in-memory Bio.PDB.Structure object.
    """
    scores = []
    # Ensure rdmols and receptors are lists for consistent processing
    if not isinstance(rdmols, list):
        rdmols = [rdmols]
    if not isinstance(receptors, list):
        receptors = [receptors] * len(rdmols)

    # A single reference ligand is used for all docking jobs in this batch
    ref_ligand_path = None
    
    # Use a context manager for the reference ligand if it's an object
    if hasattr(ref_ligand_file, 'GetNumAtoms'):
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=True, mode='w+') as ref_tmp:
            Chem.MolToMolFile(ref_ligand_file, ref_tmp.name)
            ref_ligand_path = ref_tmp.name
            scores = _smina_score_worker(rdmols, receptors, ref_ligand_path, local_opt)
    else:
        # If it's already a path, use it directly
        ref_ligand_path = str(ref_ligand_file)
        scores = _smina_score_worker(rdmols, receptors, ref_ligand_path, local_opt)
        
    return scores


def _smina_score_worker(rdmols, receptors, ref_ligand_path, local_opt):
    """Worker function to perform docking, separated for cleaner file handling."""
    scores = []
    # Process each molecule-receptor pair
    for mol, receptor_input in zip(rdmols, receptors):
        try:
            # Check if the receptor is an in-memory Bio.PDB object
            if isinstance(receptor_input, Structure.Structure):
                # It's an object, so we must write it to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=True, mode='w+') as receptor_tmp:
                    io = PDBIO()
                    io.set_structure(receptor_input)
                    io.save(receptor_tmp.name)
                    receptor_filepath = receptor_tmp.name
                    
                    # Write the generated ligand to its own temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=True) as lig_tmp:
                        # --- FIX START ---
                        writer = Chem.SDWriter(lig_tmp.name) # Use .name attribute
                        writer.write(mol)
                        writer.close() # Ensure data is written before smina is called
                        # --- FIX END ---
                        
                        score_list = calculate_smina_score(
                            receptor_filepath, lig_tmp.name, ref_ligand_path, local_opt
                        )
                        scores.extend(score_list)
            else:
                # It's already a file path
                receptor_filepath = str(receptor_input)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=True) as lig_tmp:
                    # --- FIX START ---
                    writer = Chem.SDWriter(lig_tmp.name) # Use .name attribute
                    writer.write(mol)
                    writer.close() # Ensure data is written before smina is called
                    # --- FIX END ---
                    
                    score_list = calculate_smina_score(
                        receptor_filepath, lig_tmp.name, ref_ligand_path, local_opt
                    )
                    scores.extend(score_list)
        except Exception as e:
            print(f"Error during smina scoring for a molecule: {e}")
            scores.append(None) # Append a placeholder for failed molecules
    return scores


def extract_affinity(stdout: str) -> float | None:
    """
    Parse smina stdout and return the best-mode affinity (kcal/mol).
    """
    m = re.search(r"Affinity:\s+([+-]?\d+(?:\.\d+)?)", stdout)
    if m:
        return float(m.group(1))

    m = re.search(r"^\s*1\s+([+-]?\d+(?:\.\d+)?)", stdout, re.MULTILINE)
    if m:
        return float(m.group(1))

    return None


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} '
             f'-f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False):

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        os.popen(f'prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        ligand_name = f'{sdf_file.stem}_{i}'
        # prepare ligand
        ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        if out_sdf_file.exists():
            with open(out_sdf_file, 'r') as f:
                scores.append(
                    min([float(x.split()[2]) for x in f.readlines()
                         if x.startswith(' VINA RESULT:')])
                )

        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

            # run QuickVina 2
            out = os.popen(
                f'qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()

            # clean up
            ligand_pdbqt_file.unlink()

            if '-----+------------+----------+----------' not in out:
                scores.append(np.nan)
                continue

            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))

            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            if out_pdbqt_file.exists():
                os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()

                # clean up
                out_pdbqt_file.unlink()

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='moad')
    args = parser.parse_args()

    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    args.out_dir.mkdir(exist_ok=True)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            receptor_name = ligand_name[:-4]
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')

        # try:
        scores, rdmols = calculate_qvina2_score(
            receptor_file, sdf_file, args.out_dir, return_rdmol=True)
        # except AttributeError as e:
        #     print(e)
        #     continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[ligand_name] = {
                'receptor': str(receptor_file),
                'ligand': str(sdf_file),
                'scores': scores,
                'rmdols': rdmols
            }

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))
