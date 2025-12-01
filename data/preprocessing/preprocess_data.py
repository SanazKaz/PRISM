#!/usr/bin/env python

"""
Creates paired ligand-free binding pockets and ligand SDF files.

This script iterates through PDB files in an input directory. For each PDB,
it uses the RCSB API to identify all non-common, biological ligands.
"""

import os
import glob
import argparse
import sys
import requests
from pathlib import Path
from rdkit import RDLogger
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


try:
    import numpy as np
    from Bio.PDB import PDBParser, PDBIO, Select
    from rdkit import Chem
except ImportError:
    print("Error: This script requires Biopython, RDKit, and NumPy.", file=sys.stderr)
    print("Please install them (e.g., 'pip install biopython rdkit numpy')")
    sys.exit(1)

# List of common crystallographic additives and artifacts to exclude by default.
EXCLUDE_LIST = {
    # Water
    'HOH', 'WAT', 'H2O', 'DOD',
    # Glycols and PEGs
    'EDO', 'PEG', 'PGE', 'PG4', 'PE8', 'PE7', '1PE', 'P6G', 'PEO',
    # Glycerol
    'GOL', 'GLY',
    # Common ions
    'SO4', 'PO4', 'CL', 'BR', 'I', 'F',
    'NA', 'MG', 'CA', 'ZN', 'FE', 'MN', 'K', 'NI', 'CU', 'CO',
    'CD', 'HG', 'SR', 'BA', 'CS', 'RB',
    # Solvents and reducing agents
    'DMS', 'DMSO', 'ACT', 'ACE', 'BME', 'DTT', 'TRS', 'EOH', 'MeOH',
    # Buffers and crystallization agents
    'CIT', 'TAR', 'MLI', 'TRS', 'EPE', 'BEZ', 'HEPES', 'MES',
    # Detergents
    'BOG', 'B3P', 'LDA', 'SDS', 'LMT', 'PLM',
    # Cryoprotectants
    'MPD', 'IPA', 'EGL',
    # Other common artifacts
    'ACY', 'NH2', 'CO3', 'NO3', 'OH', 'O', 'NO2',
    'BCN', 'AZI', 'SCN', 'CYN', 'OCT', 'UNL', 'UNX', 'CLR', 'J40', 'NAG'
}

class PocketSelect(Select):
    def __init__(self, residues_to_keep):
        self.residues_to_keep = set(residues_to_keep)

    def accept_residue(self, residue):
        # We accept a residue if it's in our set of "keeper" residues
        return residue in self.residues_to_keep


def extract_pocket_biopython(structure, sdf_path, distance_cutoff, output_pdb_path):
    """
    Extracts and saves a pocket PDB using a pre-loaded Biopython structure and RDKit.
    """
    # 1. Load Ligand and get coordinates
    try:
        # Use MolFromMolFile for simplicity
        ligand = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
        if ligand is None:
            raise ValueError("RDKit could not read SDF file.")
        lig_coords = ligand.GetConformer(0).GetPositions()
    except Exception as e:
        raise Exception(f"Failed to read/process ligand SDF {sdf_path.name}: {e}")

    # 2. Find pocket residues (Using pre-loaded structure)
    pocket_residues = []
    
    # We iterate over the model (usually model 0)
    model = structure[0]
    
    for residue in model.get_residues():
        # This check skips HETATMs (water, EDO, etc.), only keeping standard residues
        if residue.id[0] != ' ':
            continue
            
        try:
            # Get coordinates for all atoms in this residue
            # Using list comprehension is slightly safer/faster than repeated appends
            atom_coords_list = [atom.get_coord() for atom in residue.get_atoms()]
            if not atom_coords_list:
                continue
            res_coords = np.array(atom_coords_list)
        except Exception:
            continue

        # 3. Calculate minimum distance
        # Use NumPy broadcasting to find all pairwise distances
        # (n_res_atoms, 1, 3) - (1, n_lig_atoms, 3) -> (n_res_atoms, n_lig_atoms, 3)
        diff = res_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2) # (n_res_atoms, n_lig_atoms)
        min_dist = np.sqrt(np.min(dist_sq))

        # 4. Keep residue if it's close enough
        if min_dist < distance_cutoff:
            pocket_residues.append(residue)

    # 5. Save the new pocket PDB
    if not pocket_residues:
        raise Exception(f"No protein atoms found within {distance_cutoff}A. Skipping.")

    io = PDBIO()
    io.set_structure(structure) # Give it the full structure
    # Use our helper class to select only the pocket residues
    io.save(str(output_pdb_path), select=PocketSelect(pocket_residues))


def create_binding_pockets(args):
    """
    Main execution function.
    """
    # --- 1. Setup Directories ---
    input_dir = Path(args.input_dir)
    
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        print(f"Using user-provided output directory: {base_output_dir.resolve()}")
    else:
        base_output_dir = Path(".data/preprocessed_data") 
        print(f"No output directory provided. Using default: {base_output_dir.resolve()}")

    sdf_output_dir = base_output_dir / "sdf_files"
    pocket_output_dir = base_output_dir / "pocket_files"
    split_file_path = base_output_dir / "all_data.txt" 
    
    sdf_output_dir.mkdir(exist_ok=True, parents=True)
    pocket_output_dir.mkdir(exist_ok=True, parents=True)
    
    successful_basenames = []
    
    print(f"\nInput PDBs will be read from: {input_dir.resolve()}")
    print(f"Output SDFs will be saved to: {sdf_output_dir.resolve()}")
    print(f"Output Pockets will be saved to: {pocket_output_dir.resolve()}")

    # --- 2. Find and Loop Over Input PDBs ---
    pdb_files = list(input_dir.glob('**/*.pdb'))
    if not pdb_files:
        print(f"Warning: No PDB files found in '{input_dir}'.", file=sys.stderr)
        return

    print(f"\nFound {len(pdb_files)} PDB files to process...")

    # Initialize parser once
    pdb_parser = PDBParser(QUIET=True)

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.lower()
        print(f"\n--- Processing {pdb_id} ---")

        # --- OPTIMIZATION: Parse Protein Structure ONCE per file ---
        try:
            structure = pdb_parser.get_structure(pdb_id, pdb_path)
        except Exception as e:
            print(f"  [FAIL] Could not parse PDB structure {pdb_id}: {e}")
            continue

        # --- 3. Find Biological Ligands via RCSB API ---
        try:
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}/"
            response = requests.get(entry_url)
            response.raise_for_status()
            data = response.json()
            
            # Check if container exists before accessing
            container = data.get("rcsb_entry_container_identifiers")
            if not container:
                print("  [SKIP] No entry identifiers found.")
                continue
                
            entity_ids = container.get("non_polymer_entity_ids")

            if not entity_ids:
                print("  No non-polymer entities found. Skipping.")
                continue

            for entity_id in entity_ids:
                entity_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
                response = requests.get(entity_url)
                response.raise_for_status()
                entity_data = response.json()
                
                comp_id = entity_data["pdbx_entity_nonpoly"]["comp_id"]
                
                if (not args.include_common) and comp_id in EXCLUDE_LIST:
                    print(f"  Skipping {comp_id} (common additive)")
                    continue
                
                print(f"  Found biological ligand: {comp_id}")
                
                asym_ids = entity_data["rcsb_nonpolymer_entity_container_identifiers"]["asym_ids"]

                # --- 4. Process Each Ligand Instance ---
                for chain in asym_ids:
                    try:
                        instance_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/{pdb_id}/{chain}"
                        response = requests.get(instance_url)
                        response.raise_for_status()
                        instance_data = response.json()
                        seq_id = instance_data["rcsb_nonpolymer_entity_instance_container_identifiers"]["auth_seq_id"]
                        
                        base_name = f"{pdb_id}_{comp_id}_{chain}_{seq_id}"

                        # --- 5. Download and Save Instance SDF ---
                        ligand_url = f"https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={seq_id}&label_asym_id={chain}&encoding=sdf"
                        response = requests.get(ligand_url)
                        response.raise_for_status()

                        if not response.content:
                            print(f"    [WARN] No SDF content for {base_name}. Skipping.")
                            continue

                        sdf_path = sdf_output_dir / f"{base_name}.sdf"
                        sdf_path.write_bytes(response.content)

                        # --- 6. Pocket Extraction ---
                        pocket_obj_name = f"{base_name}_pocket"
                        output_pdb_path = pocket_output_dir / f"{pocket_obj_name}.pdb"
                        
                        # Pass the ALREADY PARSED structure object
                        extract_pocket_biopython(
                            structure=structure,
                            sdf_path=sdf_path,
                            distance_cutoff=args.distance,
                            output_pdb_path=output_pdb_path
                        )
                        
                        print(f"    [OK] Saved Pocket & Ligand: {base_name}")
                        successful_basenames.append(base_name)
                        
                    except Exception as e:
                        print(f"    [FAIL] Error processing instance {pdb_id}_{comp_id}_{chain}: {e}", file=sys.stderr)

        except Exception as e:
            print(f"  [FAIL] Failed to process PDB {pdb_id} API data: {e}", file=sys.stderr)

    print("\nDone! Pocket and ligand processing complete.")
    
    # --- Write split file ---
    if successful_basenames:
        try:
            with open(split_file_path, 'w') as f:
                for name in successful_basenames:
                    f.write(name + '\n')
            print(f"\nSuccessfully generated split file: {split_file_path.resolve()}")
        except Exception as e:
            print(f"    [FAIL] Error saving split file: {e}", file=sys.stderr)
    else:
        print("\nNo successful basenames were found.")
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create paired ligand-free pocket PDBs and ligand SDFs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing the original PDB files."
    )
    
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        required=False, 
        default=None,
        help="Main directory to save all outputs."
    )

    parser.add_argument(
        "-d", "--distance",
        type=float,
        default=15.0,
        help="Distance (in Angstroms) from the ligand to define the binding pocket."
    )

    # --- FIX: Changed parser.add.argument to parser.add_argument ---
    parser.add_argument(
        "--include_common",
        action="store_true",
        help="Include pockets for common additives (e.g., EDO, SO4) that are normally excluded."
    )
    
    args = parser.parse_args()
    create_binding_pockets(args)