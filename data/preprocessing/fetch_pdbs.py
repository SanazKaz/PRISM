#!/usr/bin/env python

"""
PDB Downloader.

1. Get PDBs from CATH FunFam ID.
2. Get PDBs from UniProt Accession ID.
"""

import requests
import argparse
import sys
import time
from pathlib import Path
from typing import Set, List

#!/usr/bin/env python

def download_pdbs(pdb_ids: Set[str], output_dir: Path):
    """
    Downloads a set of PDB IDs from RCSB to the specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(pdb_ids)} PDB files to: {output_dir.resolve()}")
    
    downloaded_count = 0
    failed_count = 0
    
    for pdb_id in sorted(pdb_ids):
        pdb_id = pdb_id.lower()
        try:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            file_response = requests.get(pdb_url)
            file_response.raise_for_status()
            
            filepath = output_dir / f"{pdb_id}.pdb"
            filepath.write_text(file_response.text)
            print(f"  [OK] Downloaded: {pdb_id}.pdb")
            downloaded_count += 1
            
            time.sleep(0.1) # Be nice to RCSB server
            
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


# --- CATH Logic ---

def get_pdbs_from_funfam(funfam_id: str, output_dir: str, version: str):
    """Query CATH, find PDBs in the family, download them."""
    
    api_url = f"http://www.cathdb.info/version/{version}/api/rest/cathtree/from_cath_id_to_depth/{funfam_id}/9"
    print(f"--- CATH: Fetching tree for {funfam_id} ---")
    
    try:
        response = requests.get(api_url, headers={'Accept': 'application/json'})
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching from CATH API: {e}", file=sys.stderr)
        return

    data = response.json()
    pdb_ids = set()
    
    def extract_domains(node):
        if isinstance(node, dict):
            if 'example_domain_id' in node and node['example_domain_id']:
                pdb_ids.add(node['example_domain_id'][:4])
            if 'children' in node:
                for child in node['children']:
                    extract_domains(child)
    
    extract_domains(data)
    
    if not pdb_ids:
        print(f"Warning: No PDB IDs found for FunFam {funfam_id}.")
        return

    print(f"Found {len(pdb_ids)} unique PDB IDs in CATH family.")
    
    # --- CHANGED: Path naming logic ---
    save_path = Path(output_dir) / f"{funfam_id}_data"
    download_pdbs(pdb_ids, save_path)


# --- UniProt Logic ---

def get_pdbs_from_uniprot(uniprot_ids: List[str], output_dir: str):
    """Query UniProt, find linked PDBs, download them."""
    
    base_url = "https://rest.uniprot.org/uniprotkb"
    
    for uid in uniprot_ids:
        uid = uid.strip()
        print(f"--- UniProt: Finding PDBs for {uid} ---")
        
        request_url = f"{base_url}/{uid}.json"
        
        try:
            response = requests.get(request_url)
            response.raise_for_status()
            data = response.json()
            
            # Extract PDB references
            found_pdbs = set()
            for ref in data.get('uniProtKBCrossReferences', []):
                if ref.get('database') == 'PDB':
                    found_pdbs.add(ref.get('id'))
            
            if not found_pdbs:
                print(f"  [INFO] No PDB structures linked to UniProt ID {uid}.")
                continue
                
            print(f"  Found {len(found_pdbs)} PDBs linked to {uid}.")
            
            # --- CHANGED: Path naming logic ---
            save_path = Path(output_dir) / f"{uid}_data"
            download_pdbs(found_pdbs, save_path)
            
        except requests.exceptions.HTTPError:
            print(f"  [FAIL] UniProt ID '{uid}' not found.")
        except Exception as e:
            print(f"  [FAIL] Error processing '{uid}': {e}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Download PDB structures via CATH FunFam OR UniProt ID."
    )
    
    parser.add_argument("--funfam_id", type=str, help="CATH FunFam ID (e.g., 3.40.710.10)")
    parser.add_argument("--cath_version", type=str, default="v4_3_0", help="CATH version")
    
    parser.add_argument("--uniprot_ids", nargs='+', help="List of UniProt IDs (e.g. P12345)")
    
    # --- CHANGED: Default output is now just "data" ---
    parser.add_argument("-o", "--output_dir", type=str, default="data", help="Output directory root")
    
    args = parser.parse_args()
    
    if args.funfam_id:
        get_pdbs_from_funfam(args.funfam_id, args.output_dir, args.cath_version)
        
    if args.uniprot_ids:
        get_pdbs_from_uniprot(args.uniprot_ids, args.output_dir)

if __name__ == "__main__":
    main()