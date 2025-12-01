#!/usr/bin/env python

"""
Main pipeline script to generate a complete dataset.

Orchestrates the 3-step process:
1.  Downloads PDBs (from CATH FunFam OR UniProt IDs).
2.  Pre-processes PDBs to extract pockets and ligands.
3.  Converts pocket/ligand pairs into a final .npz dataset.
"""

import argparse
import sys
from pathlib import Path

# --- Import your 3 pipeline scripts ---
# Ensure these match your actual filenames/folder structure!
from data.preprocessing import (fetch_pdbs, preprocess_data, create_dataset)

def main():
    parser = argparse.ArgumentParser(
        description="Full data pipeline from IDs to final dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Input Data Options (Mutually Exclusive) ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cath_id",
        type=str,
        help="The full CATH FunFam ID to process (e.g., '3.40.710.10')."
    )
    group.add_argument(
        "--uniprot_ids",
        nargs='+',
        help="List of UniProt Accession IDs (e.g. P12345 Q9JIAL)."
    )
    
    # --- Optional Pipeline Parameters ---
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Main root directory for outputs. If None, creates one based on input ID."
    )
    
    # Step 1 Params
    parser.add_argument(
        "--cath_version",
        type=str,
        default="v4_3_0",
        help="CATH database version (only used if --cath_id is provided)."
    )
    
    # Step 2 Params
    parser.add_argument(
        "--preprocess_distance",
        type=float,
        default=15.0,
        help="Distance (A) to extract raw pocket environment."
    )
    parser.add_argument(
        "--include_common",
        action="store_true",
        help="Include common additives (e.g., EDO, SO4) in preprocessing."
    )
    
    # Step 3 Params
    parser.add_argument(
        "--dataset_distance",
        type=float,
        default=5.0, 
        help="Distance (A) to define final pocket residues for the model."
    )
    # Deduplication: Default ON (True), use flag to turn OFF (False)
    parser.add_argument(
        '--keep_duplicates',
        action='store_false', 
        dest='deduplicate',
        help='Use this flag to keep all duplicate instances (disables deduplication).'
    )
    parser.set_defaults(deduplicate=True)
    
    parser.add_argument(
        "--dataset_info_key",
        type=str,
        default="crossdock_full",
        help="Key from constants.py to use for encoders."
    )
    
    args = parser.parse_args()

    # --- 1. Setup Directory Structure & Naming ---
    
    # Determine Job Name
    if args.cath_id:
        job_name = f"cath_{args.cath_id.replace('.', '_')}"
        target_id_list = [args.cath_id] # Just one ID for CATH
    else:
        # For UniProt, if multiple IDs, name folder after the first one + count
        if len(args.uniprot_ids) == 1:
            job_name = f"uniprot_{args.uniprot_ids[0]}"
        else:
            job_name = f"uniprot_batch_{args.uniprot_ids[0]}_plus_{len(args.uniprot_ids)-1}"
        target_id_list = args.uniprot_ids

    # Set Output Directory
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(f"data/{job_name}_data")

    pdb_dir = base_dir / "01_raw_pdbs"
    preprocess_dir = base_dir / "02_preprocessed"
    final_dataset_dir = base_dir / "03_final_dataset"
    
    base_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print(f" Starting Pipeline: {job_name}")
    print(f" Output Directory:  {base_dir.resolve()}")
    print("="*80)

    # --- 2. Run STEP 1: Get PDBs ---
    print("\n[STEP 1/3] Fetching PDBs...")
    try:
        if args.cath_id:
            # Call CATH function
            fetch_pdbs.get_pdbs_from_funfam(
                funfam_id=args.cath_id,
                output_dir=str(pdb_dir), # Note: fetch_pdbs creates a subfolder, check logic!
                version=args.cath_version
            )
        elif args.uniprot_ids:
            # Call UniProt function
            fetch_pdbs.get_pdbs_from_uniprot(
                uniprot_ids=args.uniprot_ids,
                output_dir=str(pdb_dir)
            )
            
        print("[STEP 1/3] ✔ PDB Download Complete.")
    except Exception as e:
        print(f"[STEP 1/3] ✘ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[STEP 2/3] Preprocessing PDBs (extracting pockets/ligands)...")
    try:
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
        dataset_args = argparse.Namespace(
            input_dir=str(preprocess_dir),
            output_dir=str(final_dataset_dir),
            split_file="all_data.txt", 
            deduplicate=args.deduplicate,
            dataset_name=job_name,
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