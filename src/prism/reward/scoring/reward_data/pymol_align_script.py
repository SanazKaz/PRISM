import pymol
from pymol import cmd
import os
from glob import glob
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem

def break_pdb_file_chains(pdb_dir, output_dir):
    pymol.finish_launching(['pymol', '-qc'])
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
    pymol.finish_launching(['pymol', '-qc'])
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"))
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

def remove_solvent_and_ions(pdb_dir, output_dir, exclude_list=EXCLUDE_LIST):
    pymol.finish_launching(['pymol', '-qc'])
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
    pymol.finish_launching(['pymol', '-qc'])
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
    pymol.finish_launching(['pymol', '-qc'])
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


# ------------------------------
# NEW FUNCTION: Add bond orders by matching first 4 letters
# ------------------------------
def apply_bond_orders_from_sdf(pdb_dir, sdf_dir, output_dir, match_len=4):
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

        sdf_name = sdf_lookup[key][0]   # first match
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

        print(f"[OK] {pdb_name} + {sdf_name} → {out_path}")


# ------------------------------
#  UPDATED MAIN PIPELINE
# ------------------------------
def process_pdb_dataset(raw_pdb_dir, output_protein_dir, output_ligand_dir, original_sdf_dir, final_sdf_output):
    pymol.finish_launching(['pymol', '-qc'])

    with tempfile.TemporaryDirectory() as tmp:
        tmp1 = os.path.join(tmp, "broken")
        tmp2 = os.path.join(tmp, "aligned")
        tmp3 = os.path.join(tmp, "cleaned")
        tmp4 = os.path.join(tmp, "ligands")

        break_pdb_file_chains(raw_pdb_dir, tmp1)
        align_all_to_first(tmp1, tmp2)
        remove_solvent_and_ions(tmp2, output_protein_dir)
        remove_protein_leave_ligand(output_protein_dir, tmp4)

        # Keep ligands only near reference
        ligand_paths = glob(os.path.join(tmp4, "*.pdb"))
        remove_ligands_far_from_reference(tmp4, output_ligand_dir, ligand_paths[0])

        # ⭐ NEW: Apply bond orders using original SDFs
        apply_bond_orders_from_sdf(output_ligand_dir, original_sdf_dir, final_sdf_output, match_len=4)


process_pdb_dataset(
    raw_pdb_dir="/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/Carb_Anh_II/01_raw_pdbs/custom_list_data",
    output_protein_dir = "/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/Carb_Anh_II/04_aligned_clean_broken_chains",
    output_ligand_dir = "/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/Carb_Anh_II/05_ligand_only_within_8A",
    original_sdf_dir = "/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/Carb_Anh_II/02_preprocessed/sdf_files",
    final_sdf_output = "/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/Carb_Anh_II/FINAL_FeatMap_SDFs"
)