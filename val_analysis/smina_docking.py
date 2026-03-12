"""
SMINA Docking Module

Handles molecular docking calculations using SMINA using --minimize.
Separate from metrics calculation to maintain single responsibility.
"""

import os
import re
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from rdkit import Chem

from src.prism.utils import center_pocket_on_ligand_com


class SminaDocking:
    """
    SMINA docking helper using --minimize mode.
    Scores are read from the output SDF <minimizedAffinity> tag
    rather than stdout, which is where smina actually writes them.
    """

    def __init__(
        self,
        smina_path: str = "/data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static",
        dataset_info: Dict = None,
        local_opt: bool = False,
        timeout: int = 60,
    ):
        self.smina_path = smina_path
        self.dataset_info = dataset_info
        self.local_opt = local_opt
        self.timeout = timeout

        if not os.path.exists(smina_path):
            print(f"[WARNING] SMINA executable not found at {smina_path}; docking disabled")

        if dataset_info:
            self.data_root = Path(dataset_info["datadir"]).parent
            self.crossdocked_dir = self.data_root / "crossdocked_pocket10"
        else:
            self.crossdocked_dir = None

    @staticmethod
    def _extract_affinity_from_sdf(sdf_path: str) -> Optional[float]:
        """
        Extract minimizedAffinity from smina --minimize output SDF.
        smina writes the score as an SDF property tag:
            > <minimizedAffinity>
            -10.25880
        """
        if not sdf_path or not os.path.exists(sdf_path):
            return None
        with open(sdf_path) as f:
            content = f.read()
        m = re.search(r"<minimizedAffinity>\s*\n\s*([+-]?\d+(?:\.\d+)?)", content)
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_affinity(stdout: str, stderr: str = "") -> Optional[float]:
        """
        Fallback: robustly extract a docking affinity from stdout/stderr text.
        Kept for compatibility but --minimize scores should come from the SDF.
        Tries, in order:
          - REMARK VINA RESULT: <score>
          - Affinity: <score>
          - index table line like "1   -7.3   ..."
          - last negative float in plausible docking range
        Returns None when no sensible score found.
        """
        text = "\n".join(filter(None, [stdout or "", stderr or ""]))

        m = re.search(r"REMARK\s+VINA\s+RESULT:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

        m = re.search(r"Affinity:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

        m = re.search(r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?)", text, flags=re.MULTILINE)
        if m:
            return float(m.group(1))

        floats = re.findall(r"([+-]?\d+\.\d+)", text)
        if floats:
            nums = [float(x) for x in floats]
            negs = [n for n in nums if n < 0.0]
            if negs:
                candidate = min(negs)
                if -500.0 < candidate < 50.0:
                    return float(candidate)
            candidate = nums[-1]
            if -500.0 < candidate < 50.0:
                return float(candidate)

        return None

    def _parse_pocket_name(self, name: str) -> Tuple[str, str]:
        """
        Parse the concatenated name string produced by the CrossDocked dataset loader.

        Input format:
            {target_folder}/{stem}_pocket10.pdb_{target_folder}/{stem}.sdf

        Example:
            POL_HV1H2_489_587_0/1fb7_A_rec_1hwr_216_lig_tt_min_0_pocket10.pdb_POL_HV1H2_489_587_0/1fb7_A_rec_1hwr_216_lig_tt_min_0.sdf

        Returns:
            (target_folder, stem) e.g. ('POL_HV1H2_489_587_0', '1fb7_A_rec_1hwr_216_lig_tt_min_0')
        """
        # Split on .pdb_ to isolate the pocket half
        pocket_half = name.split(".pdb_")[0]   # e.g. POL_HV1H2_489_587_0/1fb7_..._pocket10
        parts = pocket_half.split("/")
        target_folder = parts[0]               # e.g. POL_HV1H2_489_587_0
        pocket_filename = parts[1]             # e.g. 1fb7_..._pocket10
        stem = pocket_filename.replace("_pocket10", "")  # e.g. 1fb7_...
        return target_folder, stem

    def dock_molecule(
        self, mol: Chem.Mol, receptor_pdb_path: str, ref_ligand_path: str
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Minimizes a molecule in the pocket using smina --minimize.
        Centers the pocket on the reference ligand COM before running.
        Score is read from the output SDF <minimizedAffinity> tag.
        """
        if not os.path.exists(self.smina_path):
            return {"score": None, "stdout": "", "stderr": "smina_executable_missing"}

        tmp_sdf = None
        tmp_pdb = None
        tmp_out = None

        try:
            # 1. Center pocket on reference ligand COM
            from Bio.PDB import PDBIO

            pocket_obj, ref_centered_mol = center_pocket_on_ligand_com(
                receptor_pdb_path, ref_ligand_path
            )

            if pocket_obj is None or ref_centered_mol is None:
                return {"score": None, "stdout": "", "stderr": "centering_failed"}

            # 2. Save centered pocket to temp PDB
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as pdb_tmp:
                tmp_pdb = pdb_tmp.name
                io = PDBIO()
                io.set_structure(pocket_obj)
                io.save(tmp_pdb)

            # 3. Save generated ligand to temp SDF
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as lig_tmp:
                tmp_sdf = lig_tmp.name
                writer = Chem.SDWriter(tmp_sdf)
                writer.write(mol)
                writer.close()

            # 4. Create temp output SDF path
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as out_tmp:
                tmp_out = out_tmp.name

            # 5. Build smina command - list form to safely handle any spaces in paths
            if self.local_opt:
                cmd = [
                    self.smina_path,
                    "-l", tmp_sdf,
                    "-r", tmp_pdb,
                    "--autobox_ligand", tmp_sdf,
                    "--autobox_add", "4",
                    "--exhaustiveness", "4",
                    "--num_modes", "1",
                    "-o", tmp_out,
                    "--quiet",
                ]
            else:
                cmd = [
                    self.smina_path,
                    "-l", tmp_sdf,
                    "-r", tmp_pdb,
                    "--minimize",
                    "--minimize_iters", "1000",
                    "--autobox_ligand", tmp_sdf,
                    "--autobox_add", "4",
                    "-o", tmp_out,
                    "--quiet",
                ]

            proc = subprocess.run(
                cmd, shell=False, capture_output=True, text=True, timeout=self.timeout
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # Read score from output SDF - this is where smina --minimize writes it
            score = self._extract_affinity_from_sdf(tmp_out)

            # Fallback to stdout parsing if SDF score not found
            if score is None:
                score = self._extract_affinity(stdout, stderr)

            return {"score": score, "stdout": stdout.strip(), "stderr": stderr.strip()}

        except subprocess.TimeoutExpired:
            return {"score": None, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            return {"score": None, "stdout": "", "stderr": f"exception:{e}"}
        finally:
            for f in [tmp_sdf, tmp_pdb, tmp_out]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass

    def dock_batch(
        self, molecules: List[Chem.Mol], names: List[str], max_failures: int = 5
    ) -> Dict[str, float]:
        """
        Run minimization over a batch of molecules.
        Returns aggregated score statistics across the batch.
        """
        if not molecules or not names:
            return self._empty_results()

        if self.crossdocked_dir is None:
            return self._empty_results()

        scores = []
        consecutive_failures = 0

        for mol, name in zip(molecules, names):
            if consecutive_failures >= max_failures:
                break

            try:
                target_folder, stem = self._parse_pocket_name(name)
                pocket_path = self.crossdocked_dir / target_folder / f"{stem}_pocket10.pdb"
                ref_ligand_path = self.crossdocked_dir / target_folder / f"{stem}.sdf"

                if not pocket_path.exists() or not ref_ligand_path.exists():
                    consecutive_failures += 1
                    continue

                res = self.dock_molecule(mol, str(pocket_path), str(ref_ligand_path))
                score = res.get("score") if isinstance(res, dict) else None

                if score is not None:
                    scores.append(score)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

            except Exception:
                consecutive_failures += 1
                continue

        if not scores:
            return self._empty_results()

        return {
            "smina_score_mean": float(np.mean(scores)),
            "smina_score_std": float(np.std(scores)),
            "smina_score_median": float(np.median(scores)),
            "smina_n_docked": len(scores),
            "smina_best_score": float(min(scores)),
        }

    @staticmethod
    def _empty_results() -> Dict[str, float]:
        """Return zeroed results dict when docking cannot be performed."""
        return {
            "smina_score_mean": 0.0,
            "smina_score_std": 0.0,
            "smina_score_median": 0.0,
            "smina_n_docked": 0,
            "smina_best_score": 0.0,
        }