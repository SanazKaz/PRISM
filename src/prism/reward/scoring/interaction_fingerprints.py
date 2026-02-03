import os
import tempfile
from pathlib import Path
from typing import List, Set, Optional

import torch
import warnings

from rdkit import Chem
import prolif as plf
from prolif.fingerprint import Fingerprint
from Bio.PDB import PDBIO, PDBParser

from src.prism.reward.scorer import BaseReward
from src.prism.utils import center_pocket_on_ligand_com

# suppress noisy warnings
# careful with prolif - using it incorrectly will crash the runs due to mem
warnings.filterwarnings("ignore")


def prepare_protein_for_prolif(prot_mol: Chem.Mol, target_chain: str = "A") -> Optional[Chem.Mol]:
    """
    Sanitize protein RDKit Mol for ProLIF:
    - add hydrogens (with coordinates)
    - ensure PDB residue info exists and set a chain ID
    Returns an RDKit Mol or None on failure.
    """
    try:
        prot_h = Chem.AddHs(prot_mol, addCoords=True)

        for atom in prot_h.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()
                info.SetResidueName("UNL")
                info.SetResidueNumber(1)
                info.SetIsHeteroAtom(False)
                info.SetName(f"X{atom.GetIdx()}")

                neighbors = atom.GetNeighbors()
                if neighbors:
                    nb_info = neighbors[0].GetPDBResidueInfo()
                    if nb_info:
                        info.SetResidueName(nb_info.GetResidueName())
                        info.SetResidueNumber(nb_info.GetResidueNumber())
                        info.SetIsHeteroAtom(nb_info.GetIsHeteroAtom())
                        info.SetName(f"H{atom.GetIdx()}")

            info.SetChainId(target_chain)
            atom.SetMonomerInfo(info)

        return prot_h
    except Exception:
        return None


def calculate_prolif_fingerprint_single_pair(
    pocket_obj,
    ligand_rdkit: Chem.Mol,
    base_name: Optional[str] = None,
) -> Set[str]:
    """
    Compute ProLIF fingerprint for a single protein-ligand pair using Fingerprint.generate.
    Returns a set of strings like "TYR109.A::Hydrophobic".
    - pocket_obj: Bio.PDB structure (as used with PDBIO.save)
    - ligand_rdkit: rdkit.Chem.Mol with conformer(s)
    """
    try:
        parts = (base_name or "").split('_')
        chain_id = parts[2] if len(parts) >= 3 else "A"
    except Exception:
        chain_id = "A"

    tmp_prot_path = None
    try:
        # dump pocket structure to temp PDB so RDKit can read it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_prot:
            tmp_prot_path = tmp_prot.name
            io = PDBIO()
            io.set_structure(pocket_obj)
            io.save(tmp_prot_path)

        # read to RDKit
        prot_rdkit = Chem.MolFromPDBFile(tmp_prot_path, removeHs=False, flavor=1)
        if prot_rdkit is None:
            prot_rdkit = Chem.MolFromPDBFile(tmp_prot_path, removeHs=False)
        if prot_rdkit is None:
            return set()

        protein_clean = prepare_protein_for_prolif(prot_rdkit, target_chain=chain_id)
        if protein_clean is None:
            return set()

        # build prolif Molecule objects
        prot_mol = plf.Molecule(protein_clean)

        if ligand_rdkit is None or ligand_rdkit.GetNumConformers() == 0:
            return set()

        ligand_with_h = Chem.AddHs(ligand_rdkit, addCoords=True)
        lig_mol = plf.Molecule(ligand_with_h)

        # create a local Fingerprint and use the single-pair API
        fp = Fingerprint(
            [
                "HBDonor",
                "HBAcceptor",
                "PiStacking",
                "Hydrophobic",
                "CationPi",
                "Anionic",
                "Cationic",
            ]
        )

        ifp = fp.generate(lig_mol, prot_mol, metadata=True)

        active_interactions: Set[str] = set()
        if ifp:
            for (_lres, pres), metadata in ifp.items():
                # metadata: dict interaction_name -> tuple(metadata_dicts)
                for interaction_name in metadata.keys():
                    # Create a human-readable protein residue string.
                    # pres often has attributes like resname, resid, chain or segid depending on ProLIF version.
                    try:
                        chain = pres.chain if hasattr(pres, "chain") else (pres.segid if hasattr(pres, "segid") else "A")
                        prot_res_str = f"{pres.resname}{pres.resid}.{chain}"
                    except Exception:
                        prot_res_str = str(pres)
                    active_interactions.add(f"{prot_res_str}::{interaction_name}")

        return active_interactions

    except Exception as e:
        # keep the message concise for logs
        print(f"[ProLIF] Error generating FP ({base_name}): {e}", flush=True)
        return set()

    finally:
        # cleanup temporary file (ignore failures)
        try:
            if tmp_prot_path and os.path.exists(tmp_prot_path):
                os.remove(tmp_prot_path)
        except Exception:
            pass


class InteractionFingerprintsReward(BaseReward):
    """
    Lazy reference-fingerprint caching; generated fingerprints computed on-the-fly.
    Keeps memory usage low by not using run_from_iterable or storing generated fps.
    """
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.reference_fps: dict[str, Set[str]] = {}
        self.checks_done = 0

        data_root = Path(self.dataset_info["datadir"]).parent
        self.pockets_dir = data_root / "02_preprocessed" / "pocket_files"
        self.sdf_dir = data_root / "02_preprocessed" / "sdf_files"

        print(f"[ProLIFp Init] Lazy mode: caching references. SDF dir: {self.sdf_dir}", flush=True)

    @property
    def name(self) -> str:
        return "interaction_fingerprints"

    def _compute_and_cache_reference(self, base_name: str) -> Set[str]:
        if base_name in self.reference_fps:
            return self.reference_fps[base_name]

        sdf_path = self.sdf_dir / f"{base_name}.sdf"
        pocket_path = self.pockets_dir / f"{base_name}_pocket.pdb"

        if not sdf_path.exists() or not pocket_path.exists():
            self.reference_fps[base_name] = set()
            print(f"[ProLIF CACHE] Missing files for {base_name}", flush=True)
            return set()

        try:
            pocket_centered, ligand_centered = center_pocket_on_ligand_com(str(pocket_path), str(sdf_path))
            if pocket_centered is None:
                self.reference_fps[base_name] = set()
                print(f"[ProLIF CACHE] Centering failed for {base_name}", flush=True)
                return set()

            fp_centered = calculate_prolif_fingerprint_single_pair(pocket_centered, ligand_centered, base_name)

            # optional sanity checks (limited)
            if fp_centered and self.checks_done < 5:
                try:
                    self._run_sanity_check(pocket_path, sdf_path, fp_centered, base_name)
                except Exception:
                    pass
                self.checks_done += 1

            self.reference_fps[base_name] = fp_centered or set()
            print(f"[ProLIF CACHE] Cached reference fingerprint for {base_name} ({len(self.reference_fps[base_name])})", flush=True)
            return self.reference_fps[base_name]

        except Exception as e:
            print(f"[ProLIF] Failed to compute reference for {base_name}: {e}", flush=True)
            self.reference_fps[base_name] = set()
            return set()

    def _compute_generated_fp_once(self, base_name: str, mol_gen: Chem.Mol) -> Set[str]:
        pocket_path = self.pockets_dir / f"{base_name}_pocket.pdb"
        
        # ADD DEBUG
        print(f"[DEBUG] Looking for pocket: {pocket_path}", flush=True)
        print(f"[DEBUG] Pocket exists: {pocket_path.exists()}", flush=True)
        
        if not pocket_path.exists():
            print(f"[DEBUG] Pocket file MISSING for {base_name}", flush=True)
            return set()
    
        
        if not pocket_path.exists():
            return set()

        # write mol_gen to temp SDF and reuse the same centering helper
        tmp_sdf_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".sdf", delete=False) as tmp_sdf:
                tmp_sdf_path = Path(tmp_sdf.name)
                writer = Chem.SDWriter(str(tmp_sdf_path))
                try:
                    writer.write(mol_gen)
                finally:
                    try:
                        writer.close()
                    except Exception:
                        pass

            pocket_centered, ligand_centered = center_pocket_on_ligand_com(str(pocket_path), str(tmp_sdf_path))
            if pocket_centered is None:
                return set()

            gen_fp = calculate_prolif_fingerprint_single_pair(pocket_centered, ligand_centered, base_name)
            return gen_fp or set()

        except Exception as e:
            print(f"[ProLIF] Failed to compute generated fp for {base_name}: {e}", flush=True)
            return set()

        finally:
            if tmp_sdf_path is not None and tmp_sdf_path.exists():
                try:
                    tmp_sdf_path.unlink()
                except Exception:
                    pass

    def _run_sanity_check(self, raw_pocket_path, raw_sdf_path, fp_centered, base_name):
        try:
            parser = PDBParser(QUIET=True)
            raw_pocket = parser.get_structure("raw", str(raw_pocket_path))
            suppl = Chem.SDMolSupplier(str(raw_sdf_path), removeHs=False)
            raw_ligand = suppl[0]
            fp_raw = calculate_prolif_fingerprint_single_pair(raw_pocket, raw_ligand, base_name)

            intersection = len(fp_centered.intersection(fp_raw))
            union = len(fp_centered.union(fp_raw))
            score = intersection / union if union > 0 else 0.0

            print(f"\n[Sanity Check] {base_name}", flush=True)
            print(f"  Centered Interactions: {len(fp_centered)}", flush=True)
            print(f"  Raw (Orig) Interactions: {len(fp_raw)}", flush=True)
            print(f"  Consistency Score: {score:.3f} {'complete match' if score==1.0 else 'mismatch'}", flush=True)

            if score < 1.0:
                print(f"  MISMATCH! Differences: {fp_centered.symmetric_difference(fp_raw)}", flush=True)
        except Exception as e:
            print(f"  [Sanity Check] Failed: {e}", flush=True)

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        dataset_info = dataset_info or self.dataset_info

        if not molecules:
            return torch.zeros(0, dtype=torch.float32)

        scores: List[float] = []
        names = kwargs.get("names", [])

        for idx, mol_gen in enumerate(molecules):
            try:
                if mol_gen is None:
                    scores.append(0.0)
                    continue

                sample_name = names[idx] if names else ""
                base_name = sample_name.split("_pocket")[0] if "_pocket" in sample_name else sample_name

                # lazily compute/cached reference fingerprint
                ref_set = self.reference_fps.get(base_name)
                if ref_set is None:
                    ref_set = self._compute_and_cache_reference(base_name)

                if not ref_set:
                    scores.append(0.0)
                    continue

                # compute generated fingerprint ON THE FLY (no caching)
                gen_set = self._compute_generated_fp_once(base_name, mol_gen)

                # NEW: Calculate interaction efficiency instead of Tanimoto
                num_heavy_atoms = mol_gen.GetNumHeavyAtoms()
                if num_heavy_atoms == 0:
                    efficiency = 0.0
                else:
                    efficiency = len(gen_set) / num_heavy_atoms
                    
                # Add +1 bonus if there's at least one interaction
                if len(gen_set) > 0:
                    efficiency += 1.0

                print(f"[ProLIF] Mol {idx} ({base_name}): Efficiency={efficiency:.3f} | Interactions={len(gen_set)} HeavyAtoms={num_heavy_atoms}", flush=True)
                scores.append(efficiency)

            except Exception as e:
                print(f"[ProLIF] Error processing molecule {idx}: {e}", flush=True)
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)