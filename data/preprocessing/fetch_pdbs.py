#!/usr/bin/env python

"""
PDB Downloader.

Downloads PDB structures from RCSB given a list of PDB IDs.
"""

import requests
import argparse
import time
from pathlib import Path
from typing import Set


def download_pdbs(pdb_ids: Set[str], output_dir: Path):
    """
    Downloads a set of PDB IDs from RCSB to the specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(pdb_ids)} PDB files to: {output_dir.resolve()}")
    
    downloaded_count = 0
    failed_count = 0
    
    for pdb_id in sorted(pdb_ids):
        pdb_id = pdb_id.strip().lower()
        if len(pdb_id) != 4:
            print(f"  [SKIP] Invalid PDB ID format: '{pdb_id}'")
            continue

        try:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            file_response = requests.get(pdb_url)
            file_response.raise_for_status()
            
            filepath = output_dir / f"{pdb_id}.pdb"
            filepath.write_text(file_response.text)
            print(f"  [OK] Downloaded: {pdb_id}.pdb")
            downloaded_count += 1
            
            time.sleep(0.1)  # Be nice to RCSB server
            
        except requests.exceptions.HTTPError:
            if file_response.status_code == 404:
                print(f"  [SKIP] No PDB file found for '{pdb_id}' on RCSB.")
            else:
                print(f"  [FAIL] Failed {pdb_id}: HTTP Error")
                failed_count += 1
        except Exception as e:
            print(f"  [FAIL] Failed {pdb_id}: {e}")
            failed_count += 1

    print(f"Batch Complete: {downloaded_count} downloaded, {failed_count} failed.\n")


def get_pdbs_from_file(file_path: str, output_dir: str):
    """
    Reads PDB IDs from a text file and downloads them.
    
    Supports comma-separated or newline-separated formats.
    """
    print(f"--- Reading PDBs from {file_path} ---")
    
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    content = path.read_text()
    
    # Handle both comma and newline separated formats
    raw_ids = content.replace('\n', ',').split(',')
    pdb_ids = {x.strip() for x in raw_ids if x.strip()}
    
    if not pdb_ids:
        print("Warning: No PDB IDs found in file.")
        return

    print(f"Found {len(pdb_ids)} PDB IDs in file.")
    download_pdbs(pdb_ids, Path(output_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Download PDB structures from RCSB."
    )
    parser.add_argument("--pdb_list", type=str, required=True, 
                        help="Path to text file containing PDB IDs")
    parser.add_argument("-o", "--output_dir", type=str, default="data/pdbs", 
                        help="Output directory")
    
    args = parser.parse_args()
    get_pdbs_from_file(args.pdb_list, args.output_dir)


if __name__ == "__main__":
    main()
