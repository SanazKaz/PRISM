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

warnings.filterwarnings("ignore")

# Interaction types tracked by the fingerprint
INTERACTION_TYPES = [
    "HBDonor",
    "HBAcceptor",
    "PiStacking",
    "Hydrophobic",
    "CationPi",
    "Anionic",
    "Cationic",
]

# Score is 1.0 at this many unique interaction types (linear below)
DIVERSITY_TARGET = 2


class InteractionFingerprintsReward(BaseReward):
    """
    Rewards generated ligands for forming diverse protein-ligand interactions.

    Scoring logic:
        - Extracts unique interaction *types* (e.g. HBDonor, PiStacking) from the
          generated molecule's ProLIF fingerprint.
        - Score = min(num_unique_types / DIVERSITY_TARGET, 1.0)  [linear, capped at 1.0]
        - Zero interactions -> score of 0.0.

    Reference fingerprint computation is available but disabled by default
    (use_reference=False). Set use_reference=True to enable Tanimoto-style
    comparison against the crystallographic ligand.
    """

    def __init__(self, dataset_info, use_reference: bool = False):
        self.dataset_info = dataset_info
        self.use_reference = use_reference
        self.reference_fps: dict[str, Set[str]] = {}
        self._sanity_check_done = False

        data_root = Path(self.dataset_info["datadir"]).parent
        self.pockets_dir = data_root / "02_preprocessed" / "pocket_files"
        self.sdf_dir = data_root / "02_preprocessed" / "sdf_files"

        print(f"[ProLIF] Initialised. use_reference={use_reference} | SDF dir: {self.sdf_dir}", flush=True)

    @property
    def name(self) -> str:
        return "interaction_fingerprints"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_protein(prot_mol: Chem.Mol, target_chain: str = "A") -> Optional[Chem.Mol]:
        """Add hydrogens and ensure PDB residue info / chain ID are set."""
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

    def _compute_fingerprint(
        self,
        pocket_obj,
        ligand_rdkit: Chem.Mol,
        base_name: Optional[str] = None,
    ) -> Set[str]:
        """
        Compute a ProLIF fingerprint for one protein-ligand pair.

        Args:
            pocket_obj: Bio.PDB structure object.
            ligand_rdkit: RDKit Mol with at least one conformer.
            base_name: Used to infer chain ID from filename convention.

        Returns:
            Set of interaction strings, e.g. {"TYR109.A::Hydrophobic"}.
        """
        try:
            parts = (base_name or "").split("_")
            chain_id = parts[2] if len(parts) >= 3 else "A"
        except Exception:
            chain_id = "A"

        tmp_prot_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
                tmp_prot_path = tmp.name
                io = PDBIO()
                io.set_structure(pocket_obj)
                io.save(tmp_prot_path)

            prot_rdkit = Chem.MolFromPDBFile(tmp_prot_path, removeHs=False, flavor=1)
            if prot_rdkit is None:
                prot_rdkit = Chem.MolFromPDBFile(tmp_prot_path, removeHs=False)
            if prot_rdkit is None:
                return set()

            protein_clean = self._prepare_protein(prot_rdkit, target_chain=chain_id)
            if protein_clean is None:
                return set()

            if ligand_rdkit is None or ligand_rdkit.GetNumConformers() == 0:
                return set()

            prot_mol = plf.Molecule(protein_clean)
            lig_mol = plf.Molecule(Chem.AddHs(ligand_rdkit, addCoords=True))

            fp = Fingerprint(INTERACTION_TYPES)
            ifp = fp.generate(lig_mol, prot_mol, metadata=True)

            active: Set[str] = set()
            if ifp:
                for (_lres, pres), metadata in ifp.items():
                    for interaction_name in metadata.keys():
                        try:
                            chain = (
                                pres.chain if hasattr(pres, "chain")
                                else getattr(pres, "segid", "A")
                            )
                            res_str = f"{pres.resname}{pres.resid}.{chain}"
                        except Exception:
                            res_str = str(pres)
                        active.add(f"{res_str}::{interaction_name}")
            return active

        except Exception as e:
            print(f"[ProLIF] FP computation failed ({base_name}): {e}", flush=True)
            return set()

        finally:
            try:
                if tmp_prot_path and os.path.exists(tmp_prot_path):
                    os.remove(tmp_prot_path)
            except Exception:
                pass

    def _run_sanity_check(self, raw_pocket_path, raw_sdf_path, fp_centered, base_name):
        """One-shot consistency check comparing centered vs raw fingerprints."""
        try:
            parser = PDBParser(QUIET=True)
            raw_pocket = parser.get_structure("raw", str(raw_pocket_path))
            raw_ligand = Chem.SDMolSupplier(str(raw_sdf_path), removeHs=False)[0]
            fp_raw = self._compute_fingerprint(raw_pocket, raw_ligand, base_name)

            intersection = len(fp_centered & fp_raw)
            union = len(fp_centered | fp_raw)
            score = intersection / union if union > 0 else 0.0

            print(f"[Sanity Check] {base_name} | centered={len(fp_centered)} raw={len(fp_raw)} consistency={score:.3f}", flush=True)
            if score < 1.0:
                print(f"  Differences: {fp_centered.symmetric_difference(fp_raw)}", flush=True)
        except Exception as e:
            print(f"[Sanity Check] Failed ({base_name}): {e}", flush=True)

    def _compute_and_cache_reference(self, base_name: str) -> Set[str]:
        """Lazily compute and cache the reference ligand fingerprint."""
        if base_name in self.reference_fps:
            return self.reference_fps[base_name]

        sdf_path = self.sdf_dir / f"{base_name}.sdf"
        pocket_path = self.pockets_dir / f"{base_name}_pocket.pdb"

        if not sdf_path.exists() or not pocket_path.exists():
            print(f"[ProLIF] Missing reference files for {base_name}", flush=True)
            self.reference_fps[base_name] = set()
            return set()

        try:
            pocket_centered, ligand_centered = center_pocket_on_ligand_com(str(pocket_path), str(sdf_path))
            if pocket_centered is None:
                self.reference_fps[base_name] = set()
                return set()

            fp = self._compute_fingerprint(pocket_centered, ligand_centered, base_name)

            if fp and not self._sanity_check_done:
                self._run_sanity_check(pocket_path, sdf_path, fp, base_name)
                self._sanity_check_done = True

            self.reference_fps[base_name] = fp or set()
            print(f"[ProLIF] Cached reference for {base_name} ({len(self.reference_fps[base_name])} interactions)", flush=True)
            return self.reference_fps[base_name]

        except Exception as e:
            print(f"[ProLIF] Failed to compute reference for {base_name}: {e}", flush=True)
            self.reference_fps[base_name] = set()
            return set()

    def _compute_generated_fp(self, base_name: str, mol_gen: Chem.Mol) -> Set[str]:
        """Compute fingerprint for a generated molecule (not cached)."""
        pocket_path = self.pockets_dir / f"{base_name}_pocket.pdb"
        if not pocket_path.exists():
            return set()

        tmp_sdf_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".sdf", delete=False) as tmp:
                tmp_sdf_path = Path(tmp.name)
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

            return self._compute_fingerprint(pocket_centered, ligand_centered, base_name) or set()

        except Exception as e:
            print(f"[ProLIF] Failed to compute generated fp for {base_name}: {e}", flush=True)
            return set()

        finally:
            if tmp_sdf_path is not None and tmp_sdf_path.exists():
                try:
                    tmp_sdf_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _diversity_score(gen_set: Set[str]) -> float:
        unique_types = {interaction.split("::")[-1] for interaction in gen_set}
        
        non_hydrophobic = unique_types - {"Hydrophobic"}
        
        if not unique_types:
            return 0.0
        if not non_hydrophobic:
            return 0.25  # hydrophobic only - small reward to not punish it entirely
        
        # reward scales with diversity beyond hydrophobic, target is 2 non-hydrophobic types
        # push the model harder to find non-hydrophobic interactions
        return min(0.25 + (len(non_hydrophobic) / 2) * 0.75, 1.0)

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

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

                if self.use_reference:
                    ref_set = self.reference_fps.get(base_name)
                    if ref_set is None:
                        ref_set = self._compute_and_cache_reference(base_name)
                    if not ref_set:
                        scores.append(0.0)
                        continue

                gen_set = self._compute_generated_fp(base_name, mol_gen)
                score = self._diversity_score(gen_set)

                unique_types = {i.split("::")[-1] for i in gen_set}
                print(
                    f"[ProLIF] Mol {idx} ({base_name}): score={score:.3f} | "
                    f"types={unique_types} | total_interactions={len(gen_set)}",
                    flush=True,
                )
                scores.append(score)

            except Exception as e:
                print(f"[ProLIF] Error processing molecule {idx}: {e}", flush=True)
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)