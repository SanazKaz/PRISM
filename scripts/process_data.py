#!/usr/bin/env python

"""
Main pipeline script to generate a complete dataset.

Orchestrates the 3-step process:
1.  Downloads PDBs from a text file list.
2.  Pre-processes PDBs to extract pockets and ligands.
3.  Converts pocket/ligand pairs into a final .npz dataset.
"""

import argparse
import sys
from pathlib import Path

from data.preprocessing import (fetch_pdbs, preprocess_data, create_dataset)

def main():
    parser = argparse.ArgumentParser(
        description="Full data pipeline.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input
    parser.add_argument("--pdb_list", required=True, 
                        help="Path to text file with PDB IDs")
    
    # Parameters
    parser.add_argument("-o", "--output_dir", help="Main root directory for outputs")
    parser.add_argument("--preprocess_distance", type=float, default=15.0, 
                        help="Pocket cutoff distance (Angstroms)")
    parser.add_argument("--include_common", action="store_true", 
                        help="Include common additives (skip block list)")
    parser.add_argument("--dataset_distance", type=float, default=5.0, 
                        help="Final pocket distance (Angstroms)")
    parser.add_argument("--dataset_info_key", default="crossdock_full", 
                        help="Key for encoders")
    
    # Deduplication
    parser.add_argument('--keep_duplicates', action='store_false', 
                        dest='deduplicate', default=True, 
                        help='Disable deduplication')

    args = parser.parse_args()

    # --- 1. Setup Directory Structure ---
    list_name = Path(args.pdb_list).stem
    job_name = f"custom_{list_name}"

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

    # --- 2. Run STEP 1: Download PDBs ---
    print("\n[STEP 1/3] Fetching PDBs...")
    try:
        fetch_pdbs.get_pdbs_from_file(
            file_path=args.pdb_list,
            output_dir=str(pdb_dir)
        )
        print("[STEP 1/3] PDB Download Complete.")
    except Exception as e:
        print(f"[STEP 1/3] FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Run STEP 2: Preprocess ---
    print("\n[STEP 2/3] Preprocessing PDBs (extracting pockets/ligands)...")
    try:
        preprocess_args = argparse.Namespace(
            input_dir=str(pdb_dir),
            output_dir=str(preprocess_dir),
            distance=args.preprocess_distance,
            include_common=args.include_common
        )
        preprocess_data.create_binding_pockets(preprocess_args)
        print("[STEP 2/3] Preprocessing Complete.")
    except Exception as e:
        print(f"[STEP 2/3] FAILED: {e}")
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
        print("[STEP 3/3] Final Dataset Created.")
    except Exception as e:
        print(f"[STEP 3/3] FAILED: {e}")
        sys.exit(1)
        
    print("\n" + "="*80)
    print("Pipeline Finished Successfully!")
    print(f"Final dataset is located in: {final_dataset_dir.resolve()}")
    print("="*80)

if __name__ == "__main__":
    main()