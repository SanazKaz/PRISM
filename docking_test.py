"""
Docking Alignment Sanity Check

Tests whether the pocket centering is correct by docking the REFERENCE ligand
(native pose) against the centered pocket. 

Expected results:
- If alignment is correct: Reference ligand should dock with excellent scores
  (typically -7 to -12 kcal/mol for drug-like molecules)
- If alignment is broken: Scores will be poor or docking will fail

Usage:
    python test_docking_alignment.py --data_root /path/to/data --n_samples 5
"""

import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO


def center_pocket_on_ligand_com(pocket_pdb_path: str, ref_ligand_sdf_path: str):
    """
    Centers a protein pocket and reference ligand based on the ligand's 
    center of mass (CoM). Rigid translation only.
    
    Returns:
        (pocket_structure, centered_ref_mol) or (None, None) on failure
    """
    # Load the reference ligand
    ref_mol_supplier = Chem.SDMolSupplier(ref_ligand_sdf_path, removeHs=False)
    ref_mol = next((m for m in ref_mol_supplier if m is not None), None)
    if ref_mol is None:
        print(f"  [ERROR] Could not load ligand from {ref_ligand_sdf_path}")
        return None, None

    # Load the pocket
    pocket_parser = PDBParser(QUIET=True)
    try:
        pocket_structure = pocket_parser.get_structure("pocket", pocket_pdb_path)
    except Exception as e:
        print(f"  [ERROR] Could not load pocket from {pocket_pdb_path}: {e}")
        return None, None

    # Calculate ligand CoM
    lig_coords = ref_mol.GetConformer(0).GetPositions()
    lig_com = np.mean(lig_coords, axis=0)

    # Center the pocket (translate by -CoM)
    for atom in pocket_structure.get_atoms():
        atom.set_coord(atom.get_coord() - lig_com)

    # Center the reference ligand (translate by -CoM)
    centered_ref_conf = Chem.Conformer(ref_mol.GetNumAtoms())
    for i in range(ref_mol.GetNumAtoms()):
        new_pos = lig_coords[i] - lig_com
        centered_ref_conf.SetAtomPosition(i, new_pos)
    
    ref_mol.RemoveAllConformers()
    ref_mol.AddConformer(centered_ref_conf)

    return pocket_structure, ref_mol


def dock_and_get_score(
    smina_path: str,
    mol: Chem.Mol,
    pocket_structure,
    score_only: bool = True,
    timeout: int = 60
) -> Tuple[Optional[float], str]:
    """
    Dock a molecule against a pocket structure.
    
    Args:
        smina_path: Path to SMINA executable
        mol: RDKit molecule to dock
        pocket_structure: Bio.PDB structure (already centered)
        score_only: If True, just score the pose. If False, do local optimization.
        timeout: Timeout in seconds
        
    Returns:
        (score, stdout) or (None, error_message)
    """
    tmp_pdb = None
    tmp_sdf = None
    
    try:
        # Save centered pocket to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            tmp_pdb = f.name
            io = PDBIO()
            io.set_structure(pocket_structure)
            io.save(tmp_pdb)
        
        # Save ligand to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            tmp_sdf = f.name
            writer = Chem.SDWriter(tmp_sdf)
            writer.write(mol)
            writer.close()
        
        # Build SMINA command
        if score_only:
            cmd = f"{smina_path} -l {tmp_sdf} -r {tmp_pdb} --score_only"
        else:
            cmd = (f"{smina_path} -l {tmp_sdf} -r {tmp_pdb} "
                   f"--autobox_ligand {tmp_sdf} --exhaustiveness 4 --num_modes 1")
        
        # Run SMINA
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        
        # Parse score from output
        import re
        
        # Try different patterns
        # Pattern 1: "Affinity: -7.3"
        m = re.search(r"Affinity:\s*([+-]?\d+(?:\.\d+)?)", stdout)
        if m:
            return float(m.group(1)), stdout
        
        # Pattern 2: "REMARK VINA RESULT: -7.3"
        m = re.search(r"REMARK\s+VINA\s+RESULT:\s*([+-]?\d+(?:\.\d+)?)", stdout)
        if m:
            return float(m.group(1)), stdout
        
        # Pattern 3: Table line "1  -7.3  0.000  0.000"
        m = re.search(r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?)", stdout, re.MULTILINE)
        if m:
            return float(m.group(1)), stdout
        
        return None, f"Could not parse score.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"Exception: {e}"
    finally:
        for f in [tmp_pdb, tmp_sdf]:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass


def run_alignment_test(
    data_root: Path,
    smina_path: str,
    n_samples: int = 5,
    score_only: bool = True
):
    """
    Run the alignment sanity check on multiple samples.
    """
    pocket_dir = data_root / "02_preprocessed" / "pocket_files"
    sdf_dir = data_root / "02_preprocessed" / "sdf_files"
    
    if not pocket_dir.exists():
        print(f"[ERROR] Pocket directory not found: {pocket_dir}")
        return
    if not sdf_dir.exists():
        print(f"[ERROR] SDF directory not found: {sdf_dir}")
        return
    
    # Find matching pocket/ligand pairs
    pocket_files = sorted(pocket_dir.glob("*_pocket.pdb"))[:n_samples]
    
    if not pocket_files:
        print("[ERROR] No pocket files found!")
        return
    
    print("=" * 70)
    print("DOCKING ALIGNMENT SANITY CHECK")
    print("=" * 70)
    print(f"Testing {len(pocket_files)} samples")
    print(f"Mode: {'score_only' if score_only else 'local_optimization'}")
    print("=" * 70)
    print()
    
    results = []
    
    for pocket_path in pocket_files:
        # Extract base name: "1abc_A_REC_pocket.pdb" -> "1abc_A_REC"
        base_name = pocket_path.stem.replace("_pocket", "")
        sdf_path = sdf_dir / f"{base_name}.sdf"
        
        print(f"[TEST] {base_name}")
        
        if not sdf_path.exists():
            print(f"  [SKIP] Reference ligand not found: {sdf_path}")
            print()
            continue
        
        # Step 1: Load original coordinates
        orig_mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
        if orig_mol is None:
            print(f"  [SKIP] Could not load ligand")
            print()
            continue
            
        orig_coords = orig_mol.GetConformer(0).GetPositions()
        orig_com = np.mean(orig_coords, axis=0)
        print(f"  Original ligand COM: ({orig_com[0]:.2f}, {orig_com[1]:.2f}, {orig_com[2]:.2f})")
        
        # Step 2: Center pocket and ligand
        pocket_centered, ligand_centered = center_pocket_on_ligand_com(
            str(pocket_path), str(sdf_path)
        )
        
        if pocket_centered is None or ligand_centered is None:
            print(f"  [SKIP] Centering failed")
            print()
            continue
        
        # Verify centering
        centered_coords = ligand_centered.GetConformer(0).GetPositions()
        centered_com = np.mean(centered_coords, axis=0)
        print(f"  Centered ligand COM: ({centered_com[0]:.2f}, {centered_com[1]:.2f}, {centered_com[2]:.2f})")
        
        # Step 3: Dock the centered reference ligand
        score, output = dock_and_get_score(
            smina_path, ligand_centered, pocket_centered, score_only=score_only
        )
        
        if score is not None:
            print(f"  Docking score: {score:.2f} kcal/mol")
            results.append((base_name, score))
            
            # Interpret the score
            if score < -6.0:
                print(f"  Status: GOOD (strong binding)")
            elif score < -4.0:
                print(f"  Status: MODERATE")
            elif score < 0.0:
                print(f"  Status: WEAK (may indicate alignment issue)")
            else:
                print(f"  Status: POSITIVE SCORE - ALIGNMENT LIKELY BROKEN")
        else:
            print(f"  [FAILED] {output[:200]}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        scores = [s for _, s in results]
        print(f"Successfully docked: {len(results)}/{len(pocket_files)}")
        print(f"Mean score: {np.mean(scores):.2f} kcal/mol")
        print(f"Best score: {min(scores):.2f} kcal/mol")
        print(f"Worst score: {max(scores):.2f} kcal/mol")
        print()
        
        # Verdict
        if np.mean(scores) < -5.0:
            print("VERDICT: Alignment appears CORRECT")
            print("  Reference ligands dock well, so if generated ligands score poorly,")
            print("  the issue is with the generated molecules, not the alignment.")
        elif np.mean(scores) < -2.0:
            print("VERDICT: Alignment QUESTIONABLE")
            print("  Scores are weaker than expected for native poses.")
            print("  Consider investigating further.")
        else:
            print("VERDICT: Alignment likely BROKEN")
            print("  Native poses should not have scores this poor.")
    else:
        print("No successful docking runs - check file paths and SMINA installation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test docking alignment")
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="Root data directory (parent of 02_preprocessed)"
    )
    parser.add_argument(
        "--smina_path",
        type=str,
        default="/data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static",
        help="Path to SMINA executable"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--local_opt",
        action="store_true",
        help="Use local optimization instead of score_only"
    )
    
    args = parser.parse_args()
    
    run_alignment_test(
        data_root=Path(args.data_root),
        smina_path=args.smina_path,
        n_samples=args.n_samples,
        score_only=not args.local_opt
    )