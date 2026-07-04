"""PoseBustersGeometryReward

Adapted from PoseBusters (Buttenschoen et al.) for Reinforcement Learning.
Uses the original PoseBusters check_geometry function directly.

Checks:
    1. Bond lengths (1-2 interactions) - within expected bounds
    2. Bond angles (1-3 interactions) - within expected bounds
    3. Steric clashes (non-bonded) - no VDW overlap

Score = exp(-penalty * scale).

Threshold-gated (metric-aligned): a bond/angle/clash contributes to the
penalty ONLY when it deviates beyond the threshold (the same threshold
PoseBusters uses to flag an outlier). Geometry that is within bounds
contributes nothing, so the reward does not push the model to over-optimise
already-passing molecules - it only rewards removing genuine violations,
which is exactly what moves the PoseBusters pass rate.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from posebusters.modules.distance_geometry import check_geometry

from src.prism.reward.scorer import BaseReward


class PoseBustersGeometryReward(BaseReward):
    """Threshold-gated PoseBusters geometry reward (original logic)."""

    # Flip to True to dump the PoseBusters output structure / column names once.
    _DEBUG = False
    _debugged = False

    def __init__(
        self,
        threshold_bad_bond_length: float = 0.2,
        threshold_bad_angle: float = 0.2,
        threshold_clash: float = 0.2,
        penalty_scale: float = 2.0,
        bond_weight: float = 1.5,
        angle_weight: float = 5.0,
        clash_weight: float = 1.0,
    ):
        super().__init__()
        self.threshold_bond = threshold_bad_bond_length
        self.threshold_angle = threshold_bad_angle
        self.threshold_clash = threshold_clash
        self.scale = penalty_scale
        self.w_bond = bond_weight
        self.w_angle = angle_weight
        self.w_clash = clash_weight

    @property
    def name(self) -> str:
        return "geometry_checks"

    @property
    def increase_weight_after_epoch(self) -> Optional[int]:
        return None

    @property
    def increased_weight_multiplier(self) -> float:
        return None

    def score_mol(self, mol: Mol) -> float:
        if mol is None:
            return 0.0

        results = check_geometry(
            mol,
            threshold_bad_bond_length=self.threshold_bond,
            threshold_bad_angle=self.threshold_angle,
            threshold_clash=self.threshold_clash,
        )

        details = results.get("details", {})
        df_bonds = details.get("bonds", pd.DataFrame())
        df_angles = details.get("angles", pd.DataFrame())
        df_clashes = details.get("clash", pd.DataFrame())

        if PoseBustersGeometryReward._DEBUG and not PoseBustersGeometryReward._debugged:
            PoseBustersGeometryReward._debugged = True
            print("\n========== GEOMETRY REWARD DEBUG ==========")
            print(f"details keys: {list(details.keys())}")
            for label, df in (("bonds", df_bonds), ("angles", df_angles), ("clash", df_clashes)):
                cols = list(df.columns) if isinstance(df, pd.DataFrame) else "N/A"
                print(f"  {label:6s} shape={getattr(df, 'shape', None)} columns={cols}")
            print("===========================================\n")

        total_penalty = 0.0

        # Bonds: percent_error is signed deviation from ideal. Only deviations
        # beyond the threshold contribute (full magnitude).
        for _, row in df_bonds.iterrows():
            bond_pen = abs(row["percent_error"])
            if bond_pen > self.threshold_bond:
                total_penalty += bond_pen * self.w_bond

        # Angles: bound_absolute_percent_error is 0 within bounds, % beyond the
        # bound outside. Only deviations beyond the threshold contribute.
        for _, row in df_angles.iterrows():
            ba_pen = row["bound_absolute_percent_error"]
            if ba_pen > self.threshold_angle:
                total_penalty += ba_pen * self.w_angle

        # Clashes: bound_percent_error is negative when too close. Only overlaps
        # beyond the threshold contribute.
        for _, row in df_clashes.iterrows():
            clash_pen = row["bound_percent_error"]
            if clash_pen < -self.threshold_clash:
                total_penalty += abs(clash_pen) * self.w_clash

        return float(np.exp(-total_penalty * self.scale))

    def __call__(self, molecules: List[Chem.Mol], dataset_info=None, **kwargs) -> torch.Tensor:
        scores = []
        for i, mol in enumerate(molecules):
            try:
                scores.append(self.score_mol(mol))
            except Exception as e:
                import traceback
                print(f"[geometry_checks] mol {i} raised {type(e).__name__}: {e}")
                traceback.print_exc()
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)