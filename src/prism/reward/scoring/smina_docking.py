import torch
from typing import List
from rdkit import Chem
from src.prism.reward.scorer import BaseReward
from val_analysis.smina_docking import SminaDocking


class SminaDockingReward(BaseReward):
    """
    SMINA Docking Reward (Raw Minimised Affinity).

    Docks each molecule into its target pocket using SMINA --minimize and
    returns the minimised affinity score, linearly normalised to [0, 1].

    Normalisation:
        score >= 0          -> 0.0  (clash / bad pose)
        score <= SCORE_MIN  -> 1.0  (excellent binding)
        between 0 and min   -> linear interpolation

    SCORE_MIN is set to -12.0 kcal/mol by default, covering the range of
    drug-like binders. Scores below this are clamped to 1.0.
    """

    SCORE_MIN: float = -12.0  # kcal/mol — mapped to reward 1.0
    SCORE_MAX: float = 0.0    # kcal/mol — mapped to reward 0.0

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
            score_min:    Score (kcal/mol) that maps to reward 1.0.
                          Adjust if your targets tend to produce stronger/weaker binders.
        """
        self.smina_path = smina_path
        self.local_opt = local_opt
        self.timeout = timeout
        self.dataset_info = dataset_info
        self.score_min = score_min
        self.docker = None  # Lazy-initialised on first call

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

    def _normalise_score(self, score: float) -> float:
        """
        Linearly maps a raw SMINA score (kcal/mol) to [0, 1].

        Scores >= 0 (clashes) return 0.0.
        Scores <= score_min return 1.0.
        """
        if score >= self.SCORE_MAX:
            return 0.0
        # Clamp at the lower bound
        score = max(score, self.score_min)
        # Linear interpolation: 0 at SCORE_MAX, 1 at score_min
        return (self.SCORE_MAX - score) / (self.SCORE_MAX - self.score_min)

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Calculate normalised docking reward for each molecule.

        Kwargs:
            dataset_info  (dict): Must contain 'datadir'.
            names         (list): Pocket name for each molecule.
            current_epoch (int):  Unused, kept for interface compatibility.

        Returns:
            torch.Tensor of shape (N,) with values in [0, 1].
        """
        dataset_info = kwargs.get("dataset_info")
        names = kwargs.get("names", [])

        if dataset_info is None:
            print("[LigandEfficiency] WARNING: No dataset_info provided, returning zeros.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        self._initialize_docker(dataset_info)

        if self.docker.pocket_dir is None or self.docker.sdf_dir is None:
            print("[LigandEfficiency] WARNING: Docking directories not available.")
            return torch.zeros(len(molecules), dtype=torch.float32)

        scores = []

        for i, mol in enumerate(molecules):
            if mol is None:
                scores.append(0.0)
                continue

            pocket_name = names[i] if i < len(names) else None
            if pocket_name is None:
                print(f"NO POCKET NAME FOUND FOR MOL {i}")
                scores.append(0.0)
                continue

            try:
                base_name = self.docker._parse_pocket_name(pocket_name)
                pocket_path = self.docker.pocket_dir / f"{base_name}_pocket.pdb"
                ref_ligand_path = self.docker.sdf_dir / f"{base_name}.sdf"
                # print(f"[DEBUG] pocket_path: {pocket_path}")
                # print(f"[DEBUG] ref_ligand_path: {ref_ligand_path}")
                # print(f"[DEBUG] pocket exists: {pocket_path.exists()}")
                # print(f"[DEBUG] ref exists: {ref_ligand_path.exists()}")
                # print(f"[DEBUG] pocket_name raw: {pocket_name}")
                # print(f"[DEBUG] base_name parsed: {base_name}")

                if not pocket_path.exists() or not ref_ligand_path.exists():
                    print(f"NO POCKET PATH OR REFERENCE LIGAND PATH FOUND FOR MOL {i}")
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

                reward = self._normalise_score(raw_score)
                print(f"[Docking] Mol {i} | Raw: {raw_score:.3f} kcal/mol | Reward: {reward:.3f}")
                scores.append(reward)

            except Exception as e:
                print(f"[LigandEfficiency] Error docking molecule {i}: {e}")
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)