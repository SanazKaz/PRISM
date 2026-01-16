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
    Downloads PDB structures from RCSB given a list of PDB IDs.
    
    Args:
        pdb_ids: Set of PDB IDs to download
        output_dir: Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pdb_id in sorted(pdb_ids):
        pdb_id = pdb_id.strip().lower()
        # Try PDB first
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            resp = requests.get(url)
            resp.raise_for_status()
            (output_dir / f"{pdb_id}.pdb").write_text(resp.text)
            print(f"  [OK] Downloaded: {pdb_id}.pdb")
        except:
            # Fallback to CIF
            try:
                url = f"https://files.rcsb.org/download/{pdb_id}.cif"
                resp = requests.get(url)
                resp.raise_for_status()
                (output_dir / f"{pdb_id}.cif").write_text(resp.text)
                print(f"  [CIF] Downloaded: {pdb_id}.cif (PDB not available)")
            except Exception as e:
                print(f"  [FAIL] {pdb_id}: No PDB or CIF found.")


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