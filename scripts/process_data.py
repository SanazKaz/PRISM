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


def filter_test_pdbs_from_split(
    all_data_path: Path,
    test_pdbs_path: Path,
    output_path: Path
) -> Path:
    """
    Filters test PDB entries from the preprocessed split file to prevent
    data leakage into the training set.

    Reads all_data.txt produced by preprocessing, removes any entry whose
    PDB ID (first token before '_') matches a test PDB ID, and writes the
    filtered list to train_data.txt.

    Args:
        all_data_path: Path to all_data.txt from preprocessing.
        test_pdbs_path: Path to test_pdbs.txt containing one PDB ID per line.
        output_path: Path to write the filtered train_data.txt.

    Returns:
        Path to the filtered output file.
    """
    test_ids = {
        line.strip().lower()
        for line in test_pdbs_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    }
    print(f"  Loaded {len(test_ids)} test PDB IDs to exclude.")

    all_entries = [
        line.strip()
        for line in all_data_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

    filtered = [e for e in all_entries if e.split('_')[0].lower() not in test_ids]
    removed = len(all_entries) - len(filtered)

    output_path.write_text('\n'.join(filtered) + '\n')
    print(f"  {len(all_entries)} total entries -> {len(filtered)} kept, {removed} removed (test set).")

    return output_path

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
                        dest='deduplicate', default=False, 
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

    # --- 3.5: Filter test PDBs from split file ---
    print("\n[STEP 2.5/3] Filtering test PDBs from training split...")
    test_pdbs_path = Path("/data/stat-cadd/wolf7055/PRISM/data/preprocessing/test_pdbs.txt")
    train_split_path = preprocess_dir / "train_data.txt"

    filter_test_pdbs_from_split(
        all_data_path=preprocess_dir / "all_data.txt",
        test_pdbs_path=test_pdbs_path,
        output_path=train_split_path
    )
    print("[STEP 2.5/3] Filtering Complete.")


    # --- 4. Run STEP 3: Create Final Dataset ---
    print("\n[STEP 3/3] Creating Final .npz Dataset...")
    try:
        dataset_args = argparse.Namespace(
            input_dir=str(preprocess_dir),
            output_dir=str(final_dataset_dir),
            split_file="train_data.txt", 
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