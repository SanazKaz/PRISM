"""
SMINA Docking Module

Handles molecular docking calculations using SMINA.
Separate from metrics calculation to maintain single responsibility.
"""

import os
import re
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from rdkit import Chem


import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rdkit import Chem

# important if using docking as a reward
from src.prism.utils import center_pocket_on_ligand_com

class SminaDocking:
    """
    SMINA docking helper — robust parsing and safer return values.
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
            self.pocket_dir = self.data_root / "02_preprocessed" / "pocket_files"
            self.sdf_dir = self.data_root / "02_preprocessed" / "sdf_files"
        else:
            self.pocket_dir = None
            self.sdf_dir = None

    @staticmethod
    def _extract_affinity(stdout: str, stderr: str = "") -> Optional[float]:
        """
        Robustly extract a docking affinity from stdout/stderr text.
        Tries, in order:
          - REMARK VINA RESULT: <score> ...
          - Affinity: <score>
          - a leading index line like "1   -7.3   ..."
          - the last numeric token on the last non-empty line (fallback)
        Returns None when no sensible score found.
        """
        text = "\n".join(filter(None, [stdout or "", stderr or ""]))

        # 1) Vina remark: "REMARK VINA RESULT:    -7.3    0.000    0.000"
        m = re.search(r"REMARK\s+VINA\s+RESULT:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

        # 2) "Affinity: -7.3"
        m = re.search(r"Affinity:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

        # 3) Lines like "   1    -7.300    0.000   0.000"
        m = re.search(r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?)", text, flags=re.MULTILINE)
        if m:
            return float(m.group(1))

        # 4) Look for any float tokens; prefer negative values (likely energies)
        floats = re.findall(r"([+-]?\d+\.\d+)", text)
        if floats:
            # convert to floats
            nums = [float(x) for x in floats]
            # prefer negative numbers in plausible docking range
            negs = [n for n in nums if n < 0.0]
            if negs:
                # choose the most negative but within sane limits
                candidate = min(negs)
                if -500.0 < candidate < 50.0:
                    return float(candidate)
            # fallback: choose last numeric token if plausible
            candidate = nums[-1]
            if -500.0 < candidate < 50.0:
                return float(candidate)

        # nothing found
        return None

    def _parse_pocket_name(self, name: str) -> str:
        if "_pocket_only.pdb" in name:
            return name.split("_pocket_only.pdb")[0]
        elif "_pocket" in name:
            return name.split("_pocket")[0]
        else:
            return name

    def dock_molecule(
        self, mol: Chem.Mol, receptor_pdb_path: str, ref_ligand_path: str
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Docks a molecule after centering the pocket to match the generated ligand's origin.
        Includes a Coordinate Check to verify alignment.
        """
        if not os.path.exists(self.smina_path):
            return {"score": None, "stdout": "", "stderr": "smina_executable_missing"}

        tmp_sdf = None
        tmp_pdb = None 
        
        try:
            # 1. ALIGNMENT: Move pocket to (0,0,0) based on the reference ligand
            from src.prism.utils import center_pocket_on_ligand_com
            from Bio.PDB import PDBIO
            from rdkit.Chem import rdMolTransforms

            # This helper should return the centered Bio.PDB object and the centered RDKit reference ligand
            pocket_obj, ref_centered_mol = center_pocket_on_ligand_com(receptor_pdb_path, ref_ligand_path)
            
            if pocket_obj is None or ref_centered_mol is None:
                return {"score": None, "stdout": "", "stderr": "centering_failed"}

            # 2. SAVE CENTERED POCKET to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as pdb_tmp:
                tmp_pdb = pdb_tmp.name
                io = PDBIO()
                io.set_structure(pocket_obj)
                io.save(tmp_pdb)

            # 3. SAVE GENERATED LIGAND (the one from the model) to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as lig_tmp:
                tmp_sdf = lig_tmp.name
                writer = Chem.SDWriter(tmp_sdf)
                writer.write(mol)
                writer.close()

            # --- COORDINATE CHECK  ---
            # gen_com = rdMolTransforms.ComputeCentroid(mol.GetConformer())
            # ref_com = rdMolTransforms.ComputeCentroid(ref_centered_mol.GetConformer())
            
            # dist = np.linalg.norm(np.array([gen_com.x, gen_com.y, gen_com.z]) - 
            #                       np.array([ref_com.x, ref_com.y, ref_com.z]))

            # print(f"\n[COORD CHECK] {Path(receptor_pdb_path).name}")
            # print(f"  Generated Ligand COM: ({gen_com.x:6.2f}, {gen_com.y:6.2f}, {gen_com.z:6.2f})")
            # print(f"  Reference Pocket COM: ({ref_com.x:6.2f}, {ref_com.y:6.2f}, {ref_com.z:6.2f})")
            # print(f"  Alignment Offset:     {dist:6.2f} Å")
            # ---------------------------------------------------------

            # 4. RUN SMINA
            # We use --autobox_ligand {tmp_sdf} because the generated ligand 
            # is now the best indicator of where the binding site origin is.
            if self.local_opt:
                cmd = (
                    f"{self.smina_path} -l {tmp_sdf} -r {tmp_pdb} "
                    f"--autobox_ligand {tmp_sdf} --exhaustiveness 4 --num_modes 1"
                )
            else:
                cmd = f"{self.smina_path} -l {tmp_sdf} -r {tmp_pdb} --score_only"

            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout)
            
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            score = self._extract_affinity(stdout, stderr)

            return {"score": score, "stdout": stdout.strip(), "stderr": stderr.strip()}

        except subprocess.TimeoutExpired:
            return {"score": None, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            return {"score": None, "stdout": "", "stderr": f"exception:{e}"}
        finally:
            # Clean up all temp files created in this call
            for f in [tmp_sdf, tmp_pdb]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    def dock_batch(self, molecules: List[Chem.Mol], names: List[str], max_failures: int = 5) -> Dict[str, float]:
        if not molecules or not names:
            return self._empty_results()

        if self.pocket_dir is None or self.sdf_dir is None:
            return self._empty_results()

        scores = []
        consecutive_failures = 0

        for mol, name in zip(molecules, names):
            if consecutive_failures >= max_failures:
                break

            try:
                base_name = self._parse_pocket_name(name)
                pocket_path = self.pocket_dir / f"{base_name}_pocket.pdb"
                ref_ligand_path = self.sdf_dir / f"{base_name}.sdf"

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
        return {
            "smina_score_mean": 0.0,
            "smina_score_std": 0.0,
            "smina_score_median": 0.0,
            "smina_n_docked": 0,
            "smina_best_score": 0.0,
        }
