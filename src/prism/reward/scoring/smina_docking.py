import torch
import numpy as np
from typing import List
from rdkit import Chem
from src.prism.reward.scorer import BaseReward
from val_analysis.smina_docking import SminaDocking, DEFAULT_SMINA_PATH


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
    USE_SQRT_NORMALISATION: bool = False

    # Linear mode bounds (kcal/mol)
    SCORE_MIN: float = -12.0   # maps to reward 1.0
    SCORE_MAX: float = 0.0     # maps to reward 0.0

    def __init__(
        self,
        smina_path: str = DEFAULT_SMINA_PATH,
        local_opt: bool = False,
        dataset_type: str = "custom",
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
        self.dataset_type = dataset_type
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
                dataset_type=self.dataset_type,
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


    def trapezoid_atom_normalisation(self, mol: Chem.Mol) -> float:
        """
        Top of trapezoid (atom count 20-40 heavy atoms = 1.0)
        left = 15-20 linear climb
        right = 40-60 linear climb
        anything off ramp = 0.0
        """

        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms < 15:
            return 0.0
        elif n_atoms < 20:
            return (n_atoms - 15) / 5
        elif n_atoms < 40:
            return 1.0
        elif n_atoms < 60:
            return (60 - n_atoms) / 20
        else:
            return 0.0



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

        # NEW
        if self.docker.crossdocked_dir is None and self.docker.pocket_dir is None:
            print("[SminaDocking] WARNING: Docking directory not available.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        scores = []

        for i, mol in enumerate(molecules):
            if mol is None:
                scores.append(0.0)
                continue

            pocket_name = names[i] if i < len(names) else None
            if pocket_name is None:
                print(f"[SminaDocking] Pocket name was: {repr(pocket_name)}")  # add this
                scores.append(0.0)
                continue
            

            try:
                pocket_path, ref_ligand_path = self.docker._resolve_paths(pocket_name)

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
                    print(f"[Docking|sqrt] Mol {i} | Raw: {raw_score:.3f} kcal/mol "
                        f"| N_atoms: {mol.GetNumHeavyAtoms()} | Reward: {reward:.3f}")
                else:
                    linear_reward = self._normalise_linear(raw_score)
                    trap_value = self.trapezoid_atom_normalisation(mol)
                    reward = linear_reward * trap_value
                    print(f"[Docking|linear+trap] Mol {i} | Raw: {raw_score:.3f} kcal/mol "
                        f"| N_atoms: {mol.GetNumHeavyAtoms()} | Linear: {linear_reward:.3f} "
                        f"| Trap: {trap_value:.3f} | Reward: {reward:.3f}")
                
                scores.append(reward)

            except Exception as e:
                print(f"[SminaDocking] Error docking molecule {i}: {e}")
                print(f"[SminaDocking] Pocket name was: {repr(pocket_name)}")  # <-- here
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)