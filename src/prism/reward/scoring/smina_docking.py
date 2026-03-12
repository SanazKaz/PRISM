import torch
import numpy as np
from typing import List
from rdkit import Chem
from src.prism.reward.scorer import BaseReward
from val_analysis.smina_docking import SminaDocking


class SminaDockingReward(BaseReward):
    """
    SMINA Docking Reward with switchable normalisation modes.

    Two normalisation modes are available via USE_SQRT_NORMALISATION flag:

    1. Linear cap (USE_SQRT_NORMALISATION = False):
       Raw score linearly mapped to [0, 1] between SCORE_MAX and SCORE_MIN.
       Known failure mode: drives toward larger molecules.

    2. PILOT-style sqrt normalisation (USE_SQRT_NORMALISATION = True):
       Raw SMINA score divided by sqrt(N_heavy_atoms). No bounds applied.
       PPO maximises this directly - more negative = better binding.
       Counteracts size inflation without collapsing to tiny molecules
       the way true ligand efficiency (/ N_atoms) does.
       Example: -12 kcal/mol with 25 heavy atoms -> -12 / sqrt(25) = -2.4

    Switch between modes by changing USE_SQRT_NORMALISATION in the class body.
    """

    # --- Toggle this flag to switch normalisation mode ---
    USE_SQRT_NORMALISATION: bool = True

    # Linear mode bounds (kcal/mol)
    SCORE_MIN: float = -12.0   # maps to reward 1.0
    SCORE_MAX: float = 0.0     # maps to reward 0.0

    def __init__(
        self,
        smina_path: str = "/data/stat-cadd/wolf7055/PRISM/val_analysis/smina.static",
        local_opt: bool = False,
        timeout: int = 45,
        dataset_info: dict = None,
        score_min: float = -12.0,
    ):
        """
        Args:
            smina_path:   Path to the SMINA executable.
            local_opt:    Use exhaustive local optimisation instead of --minimize.
            timeout:      Per-molecule docking timeout in seconds.
            dataset_info: Dict containing at minimum a 'datadir' key.
            score_min:    Score (kcal/mol) that maps to reward 1.0 in linear mode.
        """
        self.smina_path = smina_path
        self.local_opt = local_opt
        self.timeout = timeout
        self.dataset_info = dataset_info
        self.score_min = score_min
        self.docker = None  # Lazy-initialised on first call

        mode = "sqrt(N_atoms)" if self.USE_SQRT_NORMALISATION else "linear cap"
        print(f"[SminaDocking] Normalisation mode: {mode}")

    @property
    def name(self) -> str:
        return "smina_docking"

    def _initialize_docker(self, dataset_info: dict) -> None:
        """Lazy-initialise SminaDocking with dataset_info on first call."""
        if self.docker is None:
            self.docker = SminaDocking(
                smina_path=self.smina_path,
                dataset_info=dataset_info,
                local_opt=self.local_opt,
                timeout=self.timeout,
            )

    def _normalise_linear(self, score: float) -> float:
        """
        Linearly maps a raw SMINA score (kcal/mol) to [0, 1].

        Scores >= 0 (clashes) return 0.0.
        Scores <= SCORE_MIN return 1.0.
        """
        if score >= self.SCORE_MAX:
            return 0.0
        score = max(score, self.score_min)
        return (self.SCORE_MAX - score) / (self.SCORE_MAX - self.score_min)

    def _normalise_sqrt(self, score: float, mol: Chem.Mol) -> float:
        """
        PILOT-style normalisation: raw SMINA score divided by sqrt(N_heavy_atoms),
        linearly mapped to [0, 1] with 2.5 as the upper bound.

        (-score / sqrt(n_atoms)) / 2.5 maps:
            0.0 kcal/mol  -> 0.0
            -12 kcal/mol, 25 atoms -> 2.4/2.5 = 0.96
            anything >= 2.5 -> clipped to 1.0
        """
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return 0.0
        reward = (-score / np.sqrt(n_atoms)) / 2.5
        # prevents pos vina scores from getting -ve.
        reward = float(np.clip(reward, 0.0, 1.0)) 

        return reward

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Calculate docking reward for each molecule.

        Kwargs:
            dataset_info  (dict): Must contain 'datadir'.
            names         (list): Pocket name for each molecule, in format:
                                  {target_folder}/{stem}_pocket10.pdb_{target_folder}/{stem}.sdf
            current_epoch (int):  Unused, kept for interface compatibility.

        Returns:
            torch.Tensor of shape (N,) with values in [0, 1] for linear mode,
            or unbounded negative floats for sqrt mode.
        """
        dataset_info = kwargs.get("dataset_info")
        names = kwargs.get("names", [])

        if dataset_info is None:
            print("[SminaDocking] WARNING: No dataset_info provided, returning zeros.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        self._initialize_docker(dataset_info)

        if self.docker.crossdocked_dir is None:
            print("[SminaDocking] WARNING: Docking directory not available.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        scores = []

        for i, mol in enumerate(molecules):
            if mol is None:
                scores.append(0.0)
                continue

            pocket_name = names[i] if i < len(names) else None
            if pocket_name is None:
                print(f"[SminaDocking] No pocket name found for mol {i}")
                scores.append(0.0)
                continue

            try:
                target_folder, stem = self.docker._parse_pocket_name(pocket_name)
                pocket_path = self.docker.crossdocked_dir / target_folder / f"{stem}_pocket10.pdb"
                ref_ligand_path = self.docker.crossdocked_dir / target_folder / f"{stem}.sdf"

                if not pocket_path.exists() or not ref_ligand_path.exists():
                    print(f"[SminaDocking] Missing pocket or reference ligand for mol {i}")
                    scores.append(0.0)
                    continue

                result = self.docker.dock_molecule(
                    mol=mol,
                    receptor_pdb_path=str(pocket_path),
                    ref_ligand_path=str(ref_ligand_path),
                )

                raw_score = result.get("score") if isinstance(result, dict) else None

                if raw_score is None:
                    scores.append(0.0)
                    continue

                if self.USE_SQRT_NORMALISATION:
                    reward = self._normalise_sqrt(raw_score, mol)
                else:
                    reward = self._normalise_linear(raw_score)

                mode_tag = "sqrt" if self.USE_SQRT_NORMALISATION else "linear"
                print(f"[Docking|{mode_tag}] Mol {i} | Raw: {raw_score:.3f} kcal/mol "
                      f"| N_atoms: {mol.GetNumHeavyAtoms()} | Reward: {reward:.3f}")
                scores.append(reward)

            except Exception as e:
                print(f"[SminaDocking] Error docking molecule {i}: {e}")
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)