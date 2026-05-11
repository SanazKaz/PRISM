#!/usr/bin/env python

"""
Main pipeline script to generate a complete dataset from PDB IDs or local PDB files.

Orchestrates a 3-step process:
  1. Download PDB structures from RCSB (skippable with --skip_fetch).
  2. Extract ligand-free binding pockets and paired ligand SDFs.
  3. Convert pocket/ligand pairs into final .npz datasets.

Quick-start examples
--------------------
From a list of PDB IDs (downloads from RCSB):
    python -m scripts.process_data --pdb_list data/example_pdbs.txt --output_dir data/my_dataset

From local PDB files you already have on disk:
    python -m scripts.process_data --skip_fetch --pdb_dir /path/to/pdbs --output_dir data/my_dataset

Reproduce the CrossDocked training set:
    python -m scripts.process_data \\
        --pdb_list data/crossdocked_train_pdbs.txt \\
        --output_dir data/crossdocked
"""

import argparse
import sys
from pathlib import Path

from src.prism.data_processing import fetch_pdbs, preprocess_data, create_dataset

# Bundled reference files that ship with the repo
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_TEST_PDBS = _REPO_ROOT / "data" / "crossdocked_test_pdbs.txt"


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
        test_pdbs_path: Path to a file containing one PDB ID per line.
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
        description="Full data pipeline: PDB IDs → processed .npz dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )

    # --- Input ---
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--pdb_list",
        help="Path to a text file of PDB IDs (comma- or newline-separated). "
             "The pipeline downloads these from RCSB.",
    )
    input_group.add_argument(
        "--skip_fetch",
        action="store_true",
        help="Skip the RCSB download step. Requires --pdb_dir pointing at a "
             "directory of existing .pdb / .cif files.",
    )
    parser.add_argument(
        "--pdb_dir",
        help="Directory of pre-downloaded .pdb/.cif files. "
             "Used when --skip_fetch is set; otherwise ignored.",
    )

    # --- Output ---
    parser.add_argument(
        "-o", "--output_dir",
        help="Root output directory. Sub-directories 01_raw_pdbs/, "
             "02_preprocessed/, and 03_final_dataset/ are created inside it.",
    )

    # --- Processing parameters ---
    parser.add_argument(
        "--preprocess_distance", type=float, default=15.0,
        help="Pocket extraction cutoff distance (Å). Residues within this "
             "radius of any ligand atom are kept.",
    )
    parser.add_argument(
        "--include_common", action="store_true",
        help="Include common crystallographic additives (skips the block list).",
    )
    parser.add_argument(
        "--dataset_distance", type=float, default=5.0,
        help="Final pocket definition distance (Å) used when building .npz files.",
    )
    parser.add_argument(
        "--dataset_info_key", default="crossdock_full",
        help="Atom/AA encoder key from src/models/diffsbdd/constants.py.",
    )

    # --- Test-set filtering ---
    parser.add_argument(
        "--test_pdbs",
        default=str(_DEFAULT_TEST_PDBS),
        help="Path to a file of PDB IDs to exclude from the training split "
             "(prevents data leakage). Pass 'none' to skip this step.",
    )

    # --- Deduplication ---
    parser.add_argument(
        "--keep_duplicates", action="store_true",
        help="Keep all ligand instances per PDB/ligand pair. "
             "By default, only one instance per pair is kept.",
    )

    # --- Model target ---
    parser.add_argument(
        "--model", choices=["diffsbdd", "targetdiff"], default="diffsbdd",
        help="Featurisation format for the final .npz dataset. "
             "'diffsbdd' (default): 10-dim element one-hots for the pocket. "
             "'targetdiff': 27-dim features (6 elements + 20 AA types + backbone flag) "
             "matching TargetDiff's FeaturizeProteinAtom exactly. "
             "Output goes to 03_final_dataset/ (diffsbdd) or "
             "03_final_dataset_targetdiff/ (targetdiff).",
    )

    args = parser.parse_args()

    # --- Validate arguments ---
    if args.skip_fetch and not args.pdb_dir:
        parser.error("--skip_fetch requires --pdb_dir")
    if not args.skip_fetch and not args.pdb_list:
        parser.error("Provide either --pdb_list (to download from RCSB) or "
                     "--skip_fetch --pdb_dir (to use local files)")

    # --- Directory structure ---
    if args.skip_fetch:
        list_name = Path(args.pdb_dir).stem
    else:
        list_name = Path(args.pdb_list).stem

    job_name = f"custom_{list_name}"

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(f"data/{job_name}_data")

    pdb_dir        = Path(args.pdb_dir) if args.skip_fetch else base_dir / "01_raw_pdbs"
    preprocess_dir = base_dir / "02_preprocessed"
    final_dir      = base_dir / (
        "03_final_dataset_targetdiff" if args.model == "targetdiff"
        else "03_final_dataset"
    )

    base_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(f" Pipeline: {job_name}")
    print(f" Output:   {base_dir.resolve()}")
    print("=" * 80)

    # --- STEP 1: Download PDBs ---
    if args.skip_fetch:
        print(f"\n[STEP 1/3] Skipped — using existing PDB files in: {pdb_dir.resolve()}")
    else:
        print("\n[STEP 1/3] Fetching PDBs from RCSB...")
        try:
            fetch_pdbs.get_pdbs_from_file(
                file_path=args.pdb_list,
                output_dir=str(pdb_dir),
            )
            print("[STEP 1/3] Download complete.")
        except Exception as e:
            print(f"[STEP 1/3] FAILED: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # --- STEP 2: Preprocess (extract pockets + ligands) ---
    print("\n[STEP 2/3] Preprocessing: extracting pockets and ligands...")
    try:
        preprocess_args = argparse.Namespace(
            input_dir=str(pdb_dir),
            output_dir=str(preprocess_dir),
            distance=args.preprocess_distance,
            include_common=args.include_common,
        )
        preprocess_data.create_binding_pockets(preprocess_args)
        print("[STEP 2/3] Preprocessing complete.")
    except Exception as e:
        print(f"[STEP 2/3] FAILED: {e}")
        sys.exit(1)

    # --- STEP 2.5: Filter test PDBs from the split ---
    train_split_path = preprocess_dir / "train_data.txt"
    if args.test_pdbs.lower() == "none":
        print("\n[STEP 2.5/3] Skipped — no test-set filtering requested.")
        # Use all_data.txt directly as the training split
        train_split_path = preprocess_dir / "all_data.txt"
    else:
        test_pdbs_path = Path(args.test_pdbs)
        if not test_pdbs_path.exists():
            print(f"[STEP 2.5/3] WARNING: test PDB file not found at {test_pdbs_path}. "
                  "Skipping filter step.")
            train_split_path = preprocess_dir / "all_data.txt"
        else:
            print("\n[STEP 2.5/3] Filtering test PDBs from training split...")
            filter_test_pdbs_from_split(
                all_data_path=preprocess_dir / "all_data.txt",
                test_pdbs_path=test_pdbs_path,
                output_path=train_split_path,
            )
            print("[STEP 2.5/3] Filtering complete.")

    # --- STEP 3: Build final .npz dataset ---
    print("\n[STEP 3/3] Creating final .npz dataset...")
    try:
        dataset_args = argparse.Namespace(
            input_dir=str(preprocess_dir),
            output_dir=str(final_dir),
            split_file=train_split_path.name,
            deduplicate=not args.keep_duplicates,
            dataset_name=job_name,
            dataset_info_key=args.dataset_info_key,
            dist_cutoff=args.dataset_distance,
            model=args.model,
        )
        create_dataset.main(dataset_args)
        print("[STEP 3/3] Dataset creation complete.")
    except Exception as e:
        print(f"[STEP 3/3] FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Pipeline finished successfully!")
    print(f"Final dataset: {final_dir.resolve()}")
    print(f"Point 'datadir' in your config at: {final_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
