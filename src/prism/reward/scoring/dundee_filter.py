import torch
import numpy as np
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from src.prism.reward.scorer import BaseReward

RDLogger.DisableLog('rdApp.*')


class DundeeScore(BaseReward):
    """
    Dundee structural alert filter score.

    Computes a score based on the number of unwanted substructures
    (from the CHEMBL Dundee filter catalog) present in a molecule.

    Score = 1.0 - (num_alerts * penalty_per_alert), clipped to [0, 1].

    Clean molecules (no alerts) score 1.0. Each structural alert
    reduces the score by penalty_per_alert (default: 0.1).
    Invalid molecules return 0.0.

    Higher values are better.

    Parameters
    ----------
    penalty_per_alert : float
        Amount to subtract from score per structural alert (default: 0.1).

    References
    ----------
    Sterling & Irwin (2015) ZINC 15 – Ligand Discovery for Everyone.
    J. Chem. Inf. Model., 55(11), 2324-2337.
    """

    def __init__(self, penalty_per_alert: float = 0.1):
        super().__init__()
        self.penalty_per_alert = penalty_per_alert
        
        # Initialize Dundee filter catalog once
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.CHEMBL_Dundee)
        self.catalog = FilterCatalog(params)

    @property
    def name(self) -> str:
        return "dundee_score"

    def _compute_score(self, mol: Chem.Mol) -> float:
        """Computes Dundee score for a single valid molecule."""
        matches = self.catalog.GetMatches(mol)
        num_alerts = len(matches)
        score = 1.0 - (num_alerts * self.penalty_per_alert)
        return float(np.clip(score, 0.0, 1.0))

    def __call__(self, molecules: List[Chem.Mol], **kwargs) -> torch.Tensor:
        """
        Compute Dundee filter scores for a batch of molecules.

        Parameters
        ----------
        molecules : List[Chem.Mol]
            Batch of RDKit molecule objects. None entries score 0.0.

        Returns
        -------
        torch.Tensor
            1D tensor of Dundee scores in [0, 1], shape (N,).
            Invalid molecules return 0.0.
        """
        scores = []
        for mol in molecules:
            if mol is None:
                scores.append(0.0)
                continue
            try:
                scores.append(self._compute_score(mol))
            except Exception:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)