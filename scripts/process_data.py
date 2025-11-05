#!/usr/bin/env python

"""
Main pipeline script to generate a complete dataset from a CATH FunFam ID.

Orchestrates the 3-step process:
1.  Downloads all PDBs from the FunFam ID.
2.  Pre-processes PDBs to extract pockets and ligands.
3.  Converts pocket/ligand pairs into a final .npz dataset.
"""

import argparse
import sys
from pathlib import Path

# --- Import your 3 pipeline scripts from the 'new_files' package ---
from data.preprocessing import (fetch_funfam_pdbs, preprocess_data, create_dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Full data pipeline from CATH FunFam ID to final dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Core User Input ---
    parser.add_argument(
        "--funfam_id",
        type=str,
        required=True,
        help="The full CATH FunFam ID to process (e.g., '3.40.710.10.20.2.1.2')."
    )
    
    # --- Optional Pipeline Parameters ---
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Main directory to store ALL pipeline outputs. "
             "If not set, a new directory will be created "
             "based on the FunFam ID (e.g., './3_40_710_10_data/')."
    )
    parser.add_argument(
        "--cath_version",
        type=str,
        default="v4_3_0",
        help="CATH database version to query."
    )
    parser.add_argument(
        "--preprocess_distance",
        type=float,
        default=15.0,
        help="Distance (A) for PyMOL to define a pocket during preprocessing."
    )
    parser.add_argument(
        "--dataset_distance",
        type=float,
        default=5.0, # must match the model's dist_cutoff
        help="Distance (A) to define final pocket residues in create_dataset."
    )
    parser.add_argument(
        "--include_common",
        action="store_true",
        help="Include common additives (e.g., EDO, SO4) in preprocessing."
    )
    parser.add_argument(
        "--deduplicate",
        default=True,
        help="De-duplicate entries to one per PDB-ligand pair in create_dataset."
    )
    parser.add_argument(
        "--dataset_info_key",
        type=str,
        default="crossdock_full",
        help="Key from constants.py (dataset_params) to use for encoders."
    )
    
    args = parser.parse_args()

    # --- 1. Setup Directory Structure ---
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        # Create a "clean" directory name from the FunFam ID
        clean_id = args.funfam_id.rsplit('.', 3)[0].replace('.', '_') # Gets the main family ID
        base_dir = Path(f"data/{clean_id}_data")

    pdb_dir = base_dir / "01_raw_pdbs"
    preprocess_dir = base_dir / "02_preprocessed"
    final_dataset_dir = base_dir / "03_final_dataset"
    
    base_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print(f" Starting Full Pipeline for FunFam ID: {args.funfam_id}")
    print(f"Main Output Directory: {base_dir.resolve()}")
    print("="*80)

    # --- 2. Run STEP 1: Get PDBs ---
    print("\n[STEP 1/3] Downloading PDBs from CATH...")
    try:
        fetch_funfam_pdbs.get_pdbs_from_funfam(
            funfam_id=args.funfam_id,
            output_dir=str(pdb_dir),
            version=args.cath_version
        )
        print("[STEP 1/3] ✔ PDB Download Complete.")
    except Exception as e:
        print(f"[STEP 1/3] ✘ FAILED: {e}")
        sys.exit(1)

    # --- 3. Run STEP 2: Preprocess PDBs ---
    print("\n[STEP 2/3]  Preprocessing PDBs (extracting pockets/ligands)...")
    try:
        # We must create a "fake" args namespace object for the script to parse
        preprocess_args = argparse.Namespace(
            input_dir=str(pdb_dir),
            output_dir=str(preprocess_dir),
            distance=args.preprocess_distance,
            include_common=args.include_common
        )
        preprocess_data.create_binding_pockets(preprocess_args)
        print("[STEP 2/3] ✔ Preprocessing Complete.")
    except Exception as e:
        print(f"[STEP 2/3] ✘ FAILED: {e}")
        sys.exit(1)

    # --- 4. Run STEP 3: Create Final Dataset ---
    print("\n[STEP 3/3] Creating Final .npz Dataset...")
    try:
        # Create another "fake" args object
        dataset_name = args.funfam_id.replace('.', '_')
        dataset_args = argparse.Namespace(
            input_dir=str(preprocess_dir),
            output_dir=str(final_dataset_dir),
            split_file="all_data.txt",  # This is the hardcoded output from Step 2
            deduplicate=args.deduplicate,
            dataset_name=dataset_name,
            dataset_info_key=args.dataset_info_key,
            dist_cutoff=args.dataset_distance
        )
        create_dataset.main(dataset_args)
        print("[STEP 3/3] ✔ Final Dataset Created.")
    except Exception as e:
        print(f"[STEP 3/3] ✘ FAILED: {e}")
        sys.exit(1)
        
    print("\n" + "="*80)
    print("✔ Pipeline Finished Successfully!")
    print(f"Final dataset is located in: {final_dataset_dir.resolve()}")
    print("="*80)

if __name__ == "__main__":
    main()