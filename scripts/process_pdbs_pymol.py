"""
PDB Preprocessing Pipeline for Pharmacophore Hotspot Generation.

This script processes raw PDB files to extract aligned ligands suitable for
DBSCAN clustering and pharmacophore hotspot analysis.

Pipeline steps:
1. Break PDB files into separate chains
2. Align all structures to the first one
3. Remove solvent and ions
4. Extract ligands (remove protein)
5. Keep only ligands within 8A of reference
6. Apply bond orders from original SDF files

Usage:
    python preprocess_pdbs.py --target_dir /path/to/target --sdf_dir /path/to/sdfs
"""

import pymol
from pymol import cmd
import os
import argparse
from glob import glob
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem


# Residues to exclude (solvents, ions, buffers, etc.)
EXCLUDE_LIST = {
    'HOH', 'WAT', 'H2O', 'DOD',
    'EDO', 'PEG', 'PGE', 'PG4', 'PE8', 'PE7', '1PE', 'P6G', 'PEO', '2PE', 'P33', 'P40',
    'GOL', 'GLY', 'PE5', 'PE6', 'PGO', 'BME', 'EOH', 'MOH', 'ETL',
    'SO4', 'PO4', 'CL', 'BR', 'I', 'F',
    'NA', 'MG', 'CA', 'ZN', 'FE', 'MN', 'K', 'NI', 'CU', 'CO',
    'CD', 'HG', 'SR', 'BA', 'CS', 'RB', 'LI', 'AL',
    'DMS', 'DMSO', 'ACT', 'ACE', 'DTT', 'TRS', 'EOH', 'MeOH',
    'CIT', 'TAR', 'MLI', 'EPE', 'BEZ', 'HEPES', 'MES', 'FMT', 'IMD', 'POP', 'ACY',
    'BOG', 'B3P', 'LDA', 'SDS', 'LMT', 'PLM',
    'MPD', 'IPA', 'EGL',
    'NH2', 'CO3', 'NO3', 'OH', 'O', 'NO2', 'NH4',
    'BCN', 'AZI', 'SCN', 'CYN', 'OCT', 'UNL', 'UNX', 'CLR', 'J40', 'NAG',
    'ATP', 'ADP', 'AMP', 'ANP', 'ACP', 'GNP', 'GSP', 'GTP', 'GDP'
}


def break_pdb_file_chains(pdb_dir, output_dir):
    """Break multi-chain PDB files into separate single-chain files."""
    os.makedirs(output_dir, exist_ok=True)
    
    new_pdb_files = []
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
    
    for pdb in pdb_paths:
        cmd.delete("all")
        cmd.load(pdb, "structure")
        
        chains = cmd.get_chains("structure")
        print(f"{os.path.basename(pdb)}: {len(chains)} chains - {chains}")
        basename = os.path.splitext(os.path.basename(pdb))[0]
        
        for chain in chains:
            sel_name = f"chain_{chain}"
            cmd.select(sel_name, f"chain {chain}")
            
            atom_count = cmd.count_atoms(sel_name)
            if atom_count > 0:
                out_path = os.path.join(output_dir, f"{basename}_chain{chain}.pdb")
                cmd.save(out_path, sel_name)
                new_pdb_files.append(out_path)
                print(f"  Saved chain {chain}: {atom_count} atoms")
            else:
                print(f"  Skipping chain {chain}: no atoms")
    
    return new_pdb_files


def align_all_to_first(pdb_dir, output_dir):
    """Align all PDB structures to the first one."""
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
    if not pdb_paths:
        print(f"WARNING: No PDB files found in {pdb_dir}")
        return []
    
    new_pdb_files = []
    failed = []
    
    cmd.delete("all")
    cmd.load(pdb_paths[0], "anchor")
    
    for pdb in pdb_paths:
        basename = os.path.basename(pdb)
        name = os.path.splitext(basename)[0]
        
        cmd.load(pdb, name)
        
        try:
            result = cmd.align(name, "anchor", quiet=1)
            print(f"Aligned {basename} - RMSD: {result[0]:.2f}, {result[1]} atoms")
            
            out_path = os.path.join(output_dir, basename)
            cmd.save(out_path, name)
            new_pdb_files.append(out_path)
        except:
            print(f"FAILED: {basename}")
            failed.append(basename)
        
        cmd.delete(name)
    
    if failed:
        print(f"\nFailed to align: {failed}")
    
    return new_pdb_files


def remove_solvent_and_ions(pdb_dir, output_dir, exclude_list=EXCLUDE_LIST):
    """Remove solvent molecules, ions, and buffer components."""
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
    new_pdb_files = []
    
    resn_selection = " or ".join([f"resn {resn}" for resn in exclude_list])
    
    for pdb in pdb_paths:
        basename = os.path.basename(pdb)
        name = os.path.splitext(basename)[0]
        
        cmd.delete("all")
        cmd.load(pdb, name)
        
        before = cmd.count_atoms(name)
        cmd.remove(resn_selection)
        after = cmd.count_atoms(name)
        
        print(f"{basename}: removed {before - after} atoms")
        
        out_path = os.path.join(output_dir, basename)
        cmd.save(out_path, name)
        new_pdb_files.append(out_path)
    
    return new_pdb_files


def remove_protein_leave_ligand(pdb_dir, output_dir):
    """Remove protein atoms, keeping only ligand molecules."""
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
    new_pdb_files = []
    
    for pdb in pdb_paths:
        basename = os.path.basename(pdb)
        name = os.path.splitext(basename)[0]
        
        cmd.delete("all")
        cmd.load(pdb, name)
        cmd.remove("polymer")
        
        atom_count = cmd.count_atoms(name)
        if atom_count > 0:
            out_path = os.path.join(output_dir, basename)
            cmd.save(out_path, name)
            new_pdb_files.append(out_path)
            print(f"{basename}: {atom_count} ligand atoms")
        else:
            print(f"{basename}: no ligand found, skipping")
    
    return new_pdb_files


def remove_ligands_far_from_reference(pdb_dir, output_dir, reference_pdb):
    """Keep only ligand atoms within 8A of the reference ligand."""
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
    new_pdb_files = []
    
    cmd.delete("all")
    cmd.load(reference_pdb, "reference")
    print(f"Reference loaded: {cmd.count_atoms('reference')} atoms")
    
    for pdb in pdb_paths:
        basename = os.path.basename(pdb)
        name = os.path.splitext(basename)[0]
        
        cmd.load(pdb, name)
        
        nearby = cmd.count_atoms(f"byres {name} within 8 of reference")
        print(f"{basename}: {nearby} atoms within 8A of reference")
        
        if nearby > 0:
            cmd.remove(f"{name} and not (byres {name} within 8 of reference)")
            out_path = os.path.join(output_dir, basename)
            cmd.save(out_path, name)
            new_pdb_files.append(out_path)
            print(f"  -> saved {cmd.count_atoms(name)} atoms")
        else:
            print(f"  -> skipping")
        
        cmd.delete(name)
    
    return new_pdb_files


def apply_bond_orders_from_sdf(pdb_dir, sdf_dir, output_dir, match_len=4):
    """
    Apply correct bond orders to PDB ligands using template SDF files.
    
    Matches PDB files to SDF files based on first 4 characters of filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build lookup table for SDFs
    sdf_lookup = {}
    for f in os.listdir(sdf_dir):
        if f.lower().endswith(".sdf"):
            key = f[:match_len]
            sdf_lookup.setdefault(key, []).append(f)

    for pdb_name in os.listdir(pdb_dir):
        if not pdb_name.lower().endswith(".pdb"):
            continue

        key = pdb_name[:match_len]
        if key not in sdf_lookup:
            print(f"[SKIP] No matching SDF for {pdb_name}")
            continue

        sdf_name = sdf_lookup[key][0]  # first match
        pdb_path = os.path.join(pdb_dir, pdb_name)
        sdf_path = os.path.join(sdf_dir, sdf_name)

        template = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if template is None:
            print(f"[ERROR] Failed to load SDF {sdf_path}")
            continue

        pdbmol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        if pdbmol is None:
            print(f"[ERROR] Failed to load PDB {pdb_path}")
            continue

        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, pdbmol)
        except Exception as e:
            print(f"[ERROR] Bond order assignment failed for {pdb_name}: {e}")
            continue

        out_path = os.path.join(output_dir, f"{key}_aligned_with_bonds.sdf")
        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()

        print(f"[OK] {pdb_name} + {sdf_name} -> {out_path}")


def process_pdb_dataset(raw_pdb_dir, output_protein_dir, output_ligand_dir, 
                        original_sdf_dir, final_sdf_output):
    """
    Run the complete PDB preprocessing pipeline.
    
    Args:
        raw_pdb_dir: Directory containing raw PDB files
        output_protein_dir: Directory for cleaned protein structures
        output_ligand_dir: Directory for extracted ligands
        original_sdf_dir: Directory containing original SDF files with bond orders
        final_sdf_output: Directory for final aligned SDFs with correct bond orders
    """
    pymol.finish_launching(['pymol', '-qc'])

    with tempfile.TemporaryDirectory() as tmp:
        tmp1 = os.path.join(tmp, "broken")
        tmp2 = os.path.join(tmp, "aligned")
        tmp4 = os.path.join(tmp, "ligands")

        print("\n" + "="*60)
        print("Step 1: Breaking PDB files into chains...")
        print("="*60)
        break_pdb_file_chains(raw_pdb_dir, tmp1)
        
        print("\n" + "="*60)
        print("Step 2: Aligning all structures to first...")
        print("="*60)
        align_all_to_first(tmp1, tmp2)
        
        print("\n" + "="*60)
        print("Step 3: Removing solvent and ions...")
        print("="*60)
        remove_solvent_and_ions(tmp2, output_protein_dir)
        
        print("\n" + "="*60)
        print("Step 4: Extracting ligands...")
        print("="*60)
        remove_protein_leave_ligand(output_protein_dir, tmp4)

        print("\n" + "="*60)
        print("Step 5: Filtering ligands by distance to reference...")
        print("="*60)
        ligand_paths = glob(os.path.join(tmp4, "*.pdb"))
        if ligand_paths:
            remove_ligands_far_from_reference(tmp4, output_ligand_dir, ligand_paths[0])
        else:
            print("WARNING: No ligands found to filter")

        print("\n" + "="*60)
        print("Step 6: Applying bond orders from original SDFs...")
        print("="*60)
        apply_bond_orders_from_sdf(output_ligand_dir, original_sdf_dir, final_sdf_output, match_len=4)
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PDB files for pharmacophore hotspot generation"
    )
    parser.add_argument(
        "--target_dir", 
        required=True,
        help="Base directory for the target (e.g., /data/.../BRD4_BD1)"
    )
    parser.add_argument(
        "--raw_pdb_subdir",
        default="01_raw_pdbs",
        help="Subdirectory containing raw PDB files (default: 01_raw_pdbs)"
    )
    parser.add_argument(
        "--sdf_subdir",
        default="02_preprocessed/sdf_files",
        help="Subdirectory containing original SDF files (default: 02_preprocessed/sdf_files)"
    )
    parser.add_argument(
        "--protein_out_subdir",
        default="04_aligned_clean_proteins",
        help="Output subdirectory for cleaned proteins (default: 04_aligned_clean_proteins)"
    )
    parser.add_argument(
        "--ligand_out_subdir",
        default="05_ligand_only_within_8A",
        help="Output subdirectory for filtered ligands (default: 05_ligand_only_within_8A)"
    )
    parser.add_argument(
        "--final_sdf_subdir",
        default="FEATURE_MAP_ALIGNED",
        help="Output subdirectory for final aligned SDFs (default: FEATURE_MAP_ALIGNED)"
    )
    
    args = parser.parse_args()
    
    # Construct full paths
    raw_pdb_dir = os.path.join(args.target_dir, args.raw_pdb_subdir)
    sdf_dir = os.path.join(args.target_dir, args.sdf_subdir)
    protein_out_dir = os.path.join(args.target_dir, args.protein_out_subdir)
    ligand_out_dir = os.path.join(args.target_dir, args.ligand_out_subdir)
    final_sdf_dir = os.path.join(args.target_dir, args.final_sdf_subdir)
    
    print(f"\nTarget directory: {args.target_dir}")
    print(f"Raw PDBs: {raw_pdb_dir}")
    print(f"Original SDFs: {sdf_dir}")
    print(f"Protein output: {protein_out_dir}")
    print(f"Ligand output: {ligand_out_dir}")
    print(f"Final SDF output: {final_sdf_dir}")
    
    # Verify input directories exist
    if not os.path.exists(raw_pdb_dir):
        raise FileNotFoundError(f"Raw PDB directory not found: {raw_pdb_dir}")
    if not os.path.exists(sdf_dir):
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
    
    process_pdb_dataset(
        raw_pdb_dir=raw_pdb_dir,
        output_protein_dir=protein_out_dir,
        output_ligand_dir=ligand_out_dir,
        original_sdf_dir=sdf_dir,
        final_sdf_output=final_sdf_dir
    )


if __name__ == "__main__":
    main()