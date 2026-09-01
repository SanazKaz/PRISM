"""
SMINA Docking Module

Handles molecular docking calculations using SMINA using --minimize.
Supports two dataset layouts:
    - "crossdock" : CrossDocked2020 raw pocket layout
    - "custom"    : Custom preprocessed layout with separate pocket/sdf dirs
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

# The smina binary ships in this directory, so resolve it relative to this
# file rather than any one machine's checkout.
DEFAULT_SMINA_PATH = str(Path(__file__).resolve().parent / "smina.static")


VALID_DATASET_TYPES = ("crossdock", "custom")


class SminaDocking:
    """
    SMINA docking helper using --minimize mode (default) or local optimisation.

    Supports two dataset path layouts, selected via `dataset_type`:

        "crossdock" — CrossDocked2020 raw layout:
                      <datadir>/../crossdocked_pocket10/<target>/<stem>_pocket10.pdb
                      <datadir>/../crossdocked_pocket10/<target>/<stem>.sdf

        "custom"    — Custom preprocessed layout:
                      <datadir>/../02_preprocessed/pocket_files/<stem>_pocket.pdb
                      <datadir>/../02_preprocessed/sdf_files/<stem>.sdf

    Scores are read from the output SDF <minimizedAffinity> tag written by smina.
    """

    def __init__(
        self,
        smina_path: str = DEFAULT_SMINA_PATH,
        dataset_info: Dict = None,
        dataset_type: str = "crossdock",
        local_opt: bool = False,
        timeout: int = 60,
    ):
        """
        Parameters
        ----------
        smina_path   : Path to the smina executable.
        dataset_info : Dataset config dict; must contain a "datadir" key.
        dataset_type : One of "crossdock" or "custom".
        local_opt    : If True, run local optimisation instead of --minimize.
        timeout      : Subprocess timeout in seconds.
        """
        if dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(
                f"dataset_type must be one of {VALID_DATASET_TYPES}, got '{dataset_type}'"
            )

        self.smina_path = smina_path
        self.dataset_info = dataset_info
        self.dataset_type = dataset_type
        self.local_opt = local_opt
        self.timeout = timeout

        if not os.path.exists(smina_path):
            print(f"[WARNING] SMINA executable not found at {smina_path}; docking disabled")

        # Resolve data directories based on layout type
        if dataset_info:
            data_root = Path(dataset_info["datadir"]).parent

            if dataset_type == "crossdock":
                self.crossdocked_dir = data_root / "crossdocked_pocket10"
                self.pocket_dir = None
                self.sdf_dir = None
            else:  # custom
                self.crossdocked_dir = None
                self.pocket_dir = data_root / "02_preprocessed" / "pocket_files"
                self.sdf_dir = data_root / "02_preprocessed" / "sdf_files"
        else:
            self.crossdocked_dir = None
            self.pocket_dir = None
            self.sdf_dir = None

    # ------------------------------------------------------------------
    # Score extraction helpers
    # ------------------------------------------------------------------

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

        Tries, in order:
          1. REMARK VINA RESULT: <score>
          2. Affinity: <score>
          3. Index table line like "1   -7.3   ..."
          4. Last negative float in a plausible docking range
        Returns None when no sensible score is found.
        """
        text = "\n".join(filter(None, [stdout or "", stderr or ""]))

        m = re.search(
            r"REMARK\s+VINA\s+RESULT:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE
        )
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
                    return candidate
            candidate = nums[-1]
            if -500.0 < candidate < 50.0:
                return candidate

        return None

    # ------------------------------------------------------------------
    # Path resolution helpers (one per layout)
    # ------------------------------------------------------------------

    def _resolve_paths_crossdocked(self, name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Resolve pocket and reference ligand paths for the CrossDocked2020 layout.

        Input name format:
            {target_folder}/{stem}_pocket10.pdb_{target_folder}/{stem}.sdf

        Returns (pocket_path, ref_ligand_path), either of which may be None
        if the files do not exist.
        """
        pocket_half = name.split(".pdb_")[0]
        parts = pocket_half.split("/")
        target_folder = parts[0]
        stem = parts[1].replace("_pocket10", "")

        pocket_path = self.crossdocked_dir / target_folder / f"{stem}_pocket10.pdb"
        ref_ligand_path = self.crossdocked_dir / target_folder / f"{stem}.sdf"
        return pocket_path, ref_ligand_path

    def _resolve_paths_preprocessed(self, name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Resolve pocket and reference ligand paths for the preprocessed layout.

        Strips known pocket suffixes from the name to recover the base stem.

        Returns (pocket_path, ref_ligand_path), either of which may be None
        if the files do not exist.
        """
        if "_pocket_only.pdb" in name:
            stem = name.split("_pocket_only.pdb")[0]
        elif "_pocket" in name:
            stem = name.split("_pocket")[0]
        else:
            stem = name

        pocket_path = self.pocket_dir / f"{stem}_pocket.pdb"
        ref_ligand_path = self.sdf_dir / f"{stem}.sdf"
        return pocket_path, ref_ligand_path

    def _resolve_paths(self, name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Dispatch path resolution to the correct layout handler.

        Returns (pocket_path, ref_ligand_path).
        """
        if self.dataset_type == "crossdock":
            return self._resolve_paths_crossdocked(name)
        else:
            return self._resolve_paths_preprocessed(name)

    # ------------------------------------------------------------------
    # Core docking
    # ------------------------------------------------------------------

    def dock_molecule(
        self, mol: Chem.Mol, receptor_pdb_path: str, ref_ligand_path: str
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Minimise a molecule in the pocket using smina --minimize.

        Centers the pocket on the reference ligand COM before running.
        Score is read from the output SDF <minimizedAffinity> tag,
        with stdout/stderr parsing as a fallback.

        Parameters
        ----------
        mol               : RDKit Mol with a 3D conformer.
        receptor_pdb_path : Path to the pocket PDB file.
        ref_ligand_path   : Path to the reference ligand SDF (used for autobox).

        Returns
        -------
        Dict with keys: "score", "stdout", "stderr".
        """
        if not os.path.exists(self.smina_path):
            return {"score": None, "stdout": "", "stderr": "smina_executable_missing"}

        tmp_sdf = tmp_pdb = tmp_out = None

        try:
            from Bio.PDB import PDBIO

            pocket_obj, ref_centered_mol = center_pocket_on_ligand_com(
                receptor_pdb_path, ref_ligand_path
            )
            if pocket_obj is None or ref_centered_mol is None:
                return {"score": None, "stdout": "", "stderr": "centering_failed"}

            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
                tmp_pdb = f.name
                io = PDBIO()
                io.set_structure(pocket_obj)
                io.save(tmp_pdb)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
                tmp_sdf = f.name
                writer = Chem.SDWriter(tmp_sdf)
                writer.write(mol)
                writer.close()

            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
                tmp_out = f.name

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
                    "--verbosity", "2",
                    "-o", tmp_out,
                    "--quiet",
                ]

            proc = subprocess.run(
                cmd, shell=False, capture_output=True, text=True, timeout=self.timeout
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""


            score = self._extract_affinity_from_sdf(tmp_out)
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

    # ------------------------------------------------------------------
    # Batch docking
    # ------------------------------------------------------------------

    def dock_batch(
        self,
        molecules: List[Chem.Mol],
        names: List[str],
        max_failures: int = 5,
    ) -> Dict[str, float]:
        """
        Run minimisation over a batch of molecules.

        Parameters
        ----------
        molecules    : List of RDKit Mols with 3D conformers.
        names        : Corresponding molecule name strings (used to resolve paths).
        max_failures : Stop early after this many consecutive failures.

        Returns
        -------
        Dict of aggregated score statistics across the batch.
        """
        if not molecules or not names:
            return self._empty_results()

        if self.dataset_type == "crossdock" and self.crossdocked_dir is None:
            return self._empty_results()
        if self.dataset_type == "custom" and (
            self.pocket_dir is None or self.sdf_dir is None
        ):
            return self._empty_results()

        scores = []
        consecutive_failures = 0

        for mol, name in zip(molecules, names):
            if consecutive_failures >= max_failures:
                break

            try:
                pocket_path, ref_ligand_path = self._resolve_paths(name)

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