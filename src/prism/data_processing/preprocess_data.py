#!/usr/bin/env python

"""
Creates paired ligand-free binding pockets and ligand SDF files.

This script iterates through PDB files in an input directory. For each PDB,
it uses the RCSB API to identify all non-common, biological ligands.
When the RCSB REST API is unreachable (e.g. on clusters with restricted
outbound HTTP), it falls back to parsing HETATM records directly from the
downloaded PDB file and fetching ideal-geometry SDFs from files.rcsb.org.
"""

import io
import os
import glob
import argparse
import sys
import requests
from pathlib import Path
from rdkit import RDLogger
from collections import defaultdict

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


try:
    import numpy as np
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    from rdkit import Chem
except ImportError:
    print("Error: This script requires Biopython, RDKit, and NumPy.", file=sys.stderr)
    print("Please install them (e.g., 'pip install biopython rdkit numpy')")
    sys.exit(1)

# data/pdb_block_list.txt lives at <repo_root>/data/pdb_block_list.txt
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_BLOCK_LIST = _REPO_ROOT / "data" / "pdb_block_list.txt"


def load_block_list(path: Path = _DEFAULT_BLOCK_LIST):
    """
    Load blocked compound IDs from an external file.

    The block list contains crystallographic additives, common ions, solvents,
    and other non-drug-like molecules to exclude from ligand extraction.

    Returns:
        set: Compound IDs to exclude during preprocessing.
    """
    if not path.exists():
        print(f"Warning: Block list not found at {path}", file=sys.stderr)
        return set()

    content = path.read_text()
    compounds = [c.strip() for c in content.replace('\n', ',').split(',')]

    return {c for c in compounds if c}


# Allowed elements for drug-like small molecules
ALLOWED_ELEMENTS = {'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}


def is_valid_small_molecule(mol):
    """
    Validate that a molecule is a drug-like small molecule.

    Criteria:
        - 3 to 55 non-hydrogen atoms
        - At least one carbon atom
        - Contains only allowed elements (H, B, C, N, O, F, P, S, Cl, Br, I)

    Args:
        mol: RDKit Mol object

    Returns:
        tuple: (is_valid, reason) where reason explains rejection if invalid
    """
    if mol is None:
        return False, "RDKit could not parse molecule"

    heavy_atoms = mol.GetNumHeavyAtoms()

    if heavy_atoms < 3:
        return False, f"too few heavy atoms ({heavy_atoms})"

    if heavy_atoms > 55:
        return False, f"too many heavy atoms ({heavy_atoms})"

    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    if not has_carbon:
        return False, "no carbon atoms"

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ALLOWED_ELEMENTS:
            return False, f"disallowed element ({symbol})"

    return True, None


class PocketSelect(Select):
    def __init__(self, residues_to_keep):
        self.residues_to_keep = set(residues_to_keep)

    def accept_residue(self, residue):
        return residue in self.residues_to_keep


def extract_pocket_biopython(structure, sdf_path, distance_cutoff, output_pdb_path):
    """
    Extracts and saves a pocket PDB using a pre-loaded Biopython structure and RDKit.
    """
    try:
        ligand = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
        if ligand is None:
            raise ValueError("RDKit could not read SDF file.")
        lig_coords = ligand.GetConformer(0).GetPositions()
    except Exception as e:
        raise Exception(f"Failed to read/process ligand SDF {sdf_path.name}: {e}")

    pocket_residues = []
    model = structure[0]

    for residue in model.get_residues():
        if residue.id[0] != ' ':
            continue

        try:
            atom_coords_list = [atom.get_coord() for atom in residue.get_atoms()]
            if not atom_coords_list:
                continue
            res_coords = np.array(atom_coords_list)
        except Exception:
            continue

        diff = res_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        min_dist = np.sqrt(np.min(dist_sq))

        if min_dist < distance_cutoff:
            pocket_residues.append(residue)

    if not pocket_residues:
        raise Exception(f"No protein atoms found within {distance_cutoff}A. Skipping.")

    io = PDBIO()
    io.set_structure(structure)
    # NOTE: Even if the source was a CIF file, PDBIO.save() writes PDB format.
    io.save(str(output_pdb_path), select=PocketSelect(pocket_residues))


def refine_pocket_list(pocket_dir, sdf_dir, successful_basenames, tolerance=20):
    """
    Identifies and removes surface ligands due to symmetry/crystallographic artifacts
    by comparing residue counts of pockets.
    """
    print(f"\n--- Refining Pockets (Tolerance: {tolerance} residues) ---")

    groups = defaultdict(list)
    for name in successful_basenames:
        parts = name.split('_')
        key = "_".join(parts[:2])
        groups[key].append(name)

    refined_list = list(successful_basenames)
    parser = PDBParser(QUIET=True)

    for key, variants in groups.items():
        if len(variants) <= 1:
            continue

        counts = []
        for name in variants:
            pdb_path = pocket_dir / f"{name}_pocket.pdb"
            try:
                struct = parser.get_structure(name, str(pdb_path))
                res_count = len(list(struct.get_residues()))
                counts.append((name, res_count))
            except Exception:
                counts.append((name, 0))

        counts.sort(key=lambda x: x[1], reverse=True)
        max_name, max_val = counts[0]

        for i in range(1, len(counts)):
            current_name, current_val = counts[i]
            diff = max_val - current_val

            if diff > tolerance:
                print(f"    [DISCARD] {current_name} ({current_val} res): Artifact of {max_name} ({max_val} res)")

                if current_name in refined_list:
                    refined_list.remove(current_name)

                pdb_del = pocket_dir / f"{current_name}_pocket.pdb"
                sdf_del = sdf_dir / f"{current_name}.sdf"
                if pdb_del.exists(): pdb_del.unlink()
                if sdf_del.exists(): sdf_del.unlink()
            else:
                print(f"    [KEEP] {current_name} ({current_val} res): Similar to {max_name} ({max_val} res)")

    return refined_list


def _extract_ligands_local(pdb_id, structure, args, block_list,
                          sdf_output_dir, pocket_output_dir, successful_basenames):
    """Fallback when data.rcsb.org/rest/v1/ is unreachable.

    Identifies ligand residues directly from the Biopython structure (HETATM
    records). Bond topology is fetched from files.rcsb.org/ligands/download/
    (confirmed accessible); actual 3D coordinates come from the Biopython
    atoms, avoiding the label_asym_id / auth_asym_id mismatch that breaks
    the models.rcsb.org URL when both sides are not on the same HPC network.
    """
    from rdkit.Chem import AllChem

    model = structure[0]
    seen = set()

    for chain in model.get_chains():
        chain_id = chain.get_id()
        for residue in chain.get_residues():
            hetatm_flag = residue.id[0]
            if hetatm_flag == ' ' or hetatm_flag == 'W':
                continue  # skip standard residues and water

            comp_id = residue.resname.strip()
            seq_id  = str(residue.id[1])
            key     = (comp_id, chain_id, seq_id)
            if key in seen:
                continue
            seen.add(key)

            if (not args.include_common) and comp_id in block_list:
                print(f"  Skipping {comp_id} (common additive)")
                continue

            try:
                # Bond topology: ideal-geometry SDF from files.rcsb.org (accessible on HPC).
                ideal_url = f"https://files.rcsb.org/ligands/download/{comp_id}_ideal.sdf"
                resp = requests.get(ideal_url, timeout=15)
                resp.raise_for_status()
                ideal_mol = Chem.MolFromMolBlock(resp.text, removeHs=True)
                if ideal_mol is None:
                    print(f"    [SKIP] {pdb_id}_{comp_id}_{chain_id}_{seq_id}: could not parse ideal SDF")
                    continue

                # Actual 3D coordinates: write just this residue's HETATM atoms as a
                # minimal PDB block and parse with RDKit, then assign bond orders from
                # the ideal template so we get correct connectivity + real coords.
                pdb_lines = []
                for i, atom in enumerate(residue.get_atoms(), start=1):
                    x, y, z = atom.get_coord()
                    aname = atom.get_name()
                    elem  = (atom.element or aname[0]).strip()
                    pdb_lines.append(
                        f"HETATM{i:5d} {aname:<4s} {comp_id:3s} {chain_id:1s}{int(seq_id):4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}  "
                    )
                pdb_lines.append("END")
                pdb_mol = Chem.MolFromPDBBlock("\n".join(pdb_lines), removeHs=True, sanitize=False)
                if pdb_mol is None:
                    print(f"    [SKIP] {pdb_id}_{comp_id}_{chain_id}_{seq_id}: could not parse residue atoms")
                    continue

                try:
                    mol = AllChem.AssignBondOrdersFromTemplate(ideal_mol, pdb_mol)
                except Exception:
                    mol = pdb_mol  # bond-order assignment failed; proceed without explicit orders

                is_valid, reason = is_valid_small_molecule(mol)
                if not is_valid:
                    print(f"    [SKIP] {pdb_id}_{comp_id}_{chain_id}_{seq_id}: {reason}")
                    continue

                print(f"  Found biological ligand: {comp_id}")
                base_name = f"{pdb_id}_{comp_id}_{chain_id}_{seq_id}"

                sdf_path = sdf_output_dir / f"{base_name}.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()

                output_pdb_path = pocket_output_dir / f"{base_name}_pocket.pdb"
                extract_pocket_biopython(
                    structure=structure,
                    sdf_path=sdf_path,
                    distance_cutoff=args.distance,
                    output_pdb_path=output_pdb_path,
                )

                print(f"    [OK] Saved Pocket & Ligand: {base_name}")
                successful_basenames.append(base_name)

            except Exception as e:
                print(f"    [FAIL] Local extraction failed for {comp_id} in {pdb_id}: {e}", file=sys.stderr)


def create_binding_pockets(args):
    """
    Main execution function.
    """
    input_dir = Path(args.input_dir)
    block_list = load_block_list()

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

    print(f"\nInput files will be read from: {input_dir.resolve()}")
    print(f"Output SDFs will be saved to: {sdf_output_dir.resolve()}")
    print(f"Output Pockets will be saved to: {pocket_output_dir.resolve()}")

    pdb_files = list(input_dir.glob('**/*.pdb')) + list(input_dir.glob('**/*.cif'))
    if not pdb_files:
        print(f"Warning: No PDB/CIF files found in '{input_dir}'.", file=sys.stderr)
        return

    print(f"\nFound {len(pdb_files)} structural files to process...")

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.lower()
        suffix = pdb_path.suffix.lower()
        print(f"\n--- Processing {pdb_id} ({suffix.upper()}) ---")

        try:
            if suffix == '.cif':
                structure = cif_parser.get_structure(pdb_id, str(pdb_path))
            else:
                structure = pdb_parser.get_structure(pdb_id, str(pdb_path))
        except Exception as e:
            print(f"  [FAIL] Could not parse structure {pdb_id}: {e}")
            continue

        try:
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}/"
            response = requests.get(entry_url)
            response.raise_for_status()
            data = response.json()

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

                if (not args.include_common) and comp_id in block_list:
                    print(f"  Skipping {comp_id} (common additive)")
                    continue

                print(f"  Found biological ligand: {comp_id}")

                asym_ids = entity_data["rcsb_nonpolymer_entity_container_identifiers"]["asym_ids"]

                for chain in asym_ids:
                    try:
                        instance_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/{pdb_id}/{chain}"
                        response = requests.get(instance_url)
                        response.raise_for_status()
                        instance_data = response.json()
                        seq_id = instance_data["rcsb_nonpolymer_entity_instance_container_identifiers"]["auth_seq_id"]

                        base_name = f"{pdb_id}_{comp_id}_{chain}_{seq_id}"

                        ligand_url = f"https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={seq_id}&label_asym_id={chain}&encoding=sdf"
                        response = requests.get(ligand_url)
                        response.raise_for_status()

                        if not response.content:
                            print(f"    [WARN] No SDF content for {base_name}. Skipping.")
                            continue

                        mol = Chem.MolFromMolBlock(response.content.decode('utf-8'), removeHs=False)
                        is_valid, reason = is_valid_small_molecule(mol)

                        if not is_valid:
                            print(f"    [SKIP] {base_name}: {reason}")
                            continue

                        sdf_path = sdf_output_dir / f"{base_name}.sdf"
                        sdf_path.write_bytes(response.content)

                        pocket_obj_name = f"{base_name}_pocket"
                        output_pdb_path = pocket_output_dir / f"{pocket_obj_name}.pdb"

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
            print(f"  [WARN] RCSB REST API unavailable for {pdb_id} ({e}); falling back to local PDB parsing...")
            _extract_ligands_local(
                pdb_id, structure, args, block_list,
                sdf_output_dir, pocket_output_dir, successful_basenames,
            )

    print("\nDone! Extraction complete.")

    if successful_basenames:
        successful_basenames = refine_pocket_list(
            pocket_output_dir,
            sdf_output_dir,
            successful_basenames,
            tolerance=20
        )

    if successful_basenames:
        try:
            with open(split_file_path, 'w') as f:
                for name in successful_basenames:
                    f.write(name + '\n')
            print(f"\nSuccessfully generated refined split file: {split_file_path.resolve()}")
        except Exception as e:
            print(f"    [FAIL] Error saving split file: {e}", file=sys.stderr)
    else:
        print("\nNo successful basenames were found.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input_dir", required=True, help="Input structural directory")
    p.add_argument("-o", "--output_dir", help="Output directory")
    p.add_argument("-d", "--distance", type=float, default=15.0, help="Pocket cutoff distance")
    p.add_argument("--include_common", action="store_true", help="Include common additives")

    create_binding_pockets(p.parse_args())
