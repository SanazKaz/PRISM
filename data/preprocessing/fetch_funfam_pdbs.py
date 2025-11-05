#!/usr/bin/env python

"""
Downloads all PDB files associated with a specific CATH FunFam ID.

This script queries the CATH API for a given FunFam ID, finds all
unique PDB IDs from the resulting family tree, and then downloads
the corresponding PDB files from the RCSB.
"""

import requests
import argparse
import sys
from pathlib import Path

def get_pdbs_from_funfam(funfam_id: str, output_dir: str, version: str):
    """
    Fetches, parses, and downloads PDBs from a CATH FunFam ID.
    
    Args:
        funfam_id: Full FunFam CATH ID (e.g., "3.40.710.10.20.2.1.2")
        output_dir: Directory to save PDB files
        version: CATH version (e.g., "v4_3_0")
    """
    
    # --- 1. Query CATH API ---
    api_url = f"http://www.cathdb.info/version/{version}/api/rest/cathtree/from_cath_id_to_depth/{funfam_id}/9"
    
    print(f"Fetching CATH tree from: {api_url}")
    try:
        response = requests.get(api_url, headers={'Accept': 'application/json'})
        response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
    except requests.exceptions.HTTPError as err:
        print(f"Error fetching from CATH API: {err}", file=sys.stderr)
        if response.status_code == 404:
            print(f"Could not find FunFam ID '{funfam_id}' for version '{version}'.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    
    data = response.json()
    
    # --- 2. Parse Tree to find PDB IDs ---
    pdb_ids = set()
    
    def extract_domains(node):
        """Recursively search the JSON tree for domain IDs."""
        if isinstance(node, dict):
            # Use example_domain_id if it exists
            if 'example_domain_id' in node:
                domain_id = node['example_domain_id']
                if domain_id:
                    pdb_id = domain_id[:4].lower() # Get 4-char PDB code
                    pdb_ids.add(pdb_id)
            
            # Recurse through children
            if 'children' in node:
                for child in node['children']:
                    extract_domains(child)
    
    extract_domains(data)
    
    if not pdb_ids:
        print(f"Warning: No PDB IDs were found for FunFam {funfam_id}.")
        return

    print(f"Found {len(pdb_ids)} unique PDB IDs in the family.")
    
    # --- 3. Download PDBs from RCSB ---
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Downloading PDB files to: {output_path.resolve()}")
    
    downloaded_count = 0
    failed_count = 0
    
    for pdb_id in sorted(pdb_ids):
        try:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            file_response = requests.get(pdb_url)
            file_response.raise_for_status()
            
            filepath = output_path / f"{pdb_id}.pdb"
            filepath.write_text(file_response.text)
            print(f"  [OK] Downloaded: {pdb_id}.pdb")
            downloaded_count += 1
        except requests.exceptions.HTTPError as http_err:
            if file_response.status_code == 404:
                print(f"  [SKIP] No PDB file found for '{pdb_id}' on RCSB.")
            else:
                print(f"  [FAIL] Failed {pdb_id}: {http_err}")
                failed_count += 1
        except Exception as e:
            print(f"  [FAIL] Failed {pdb_id}: {e}")
            failed_count += 1
    
    print("\n--- Download Complete ---")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Skipped (Not Found):   {len(pdb_ids) - downloaded_count - failed_count}")
    print(f"Failed:                {failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Downloads PDB files for a given CATH FunFam ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--funfam_id",
        type=str,
        help="The full CATH FunFam ID to query (e.g., '3.40.710.10.20.2.1.2')"
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./cath_pdbs",
        help="Directory to save the downloaded PDB files."
    )
    
    parser.add_argument(
        "-v", "--version",
        type=str,
        default="v4_4_0",
        help="The CATH database version to query."
    )
    
    args = parser.parse_args()
    
    get_pdbs_from_funfam(args.funfam_id, args.output_dir, args.version)


if __name__ == "__main__":
    main()