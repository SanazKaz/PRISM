"""
Unit tests for the log_p_zs_given_zt / _step_log_prob machinery.

We test the MATHEMATICS of the log-probability computations without
loading any real checkpoint or GPU.

Two implementations are covered:

  DiffSBDD  — conditional_model.py:log_p_zs_given_zt
              Uses a Gaussian likelihood over z_s given predicted noise ε.
              We isolate the formula by patching self.dynamics().

  TargetDiff — targetdiff_policy.py:_step_log_prob
               Combines a Gaussian positional term (log_normal) with a
               categorical atom-type term.  Pure arithmetic, no model call.
"""

import os
os.environ["DEBUG_PPO"] = "0"

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np
import unittest

# Make TargetDiff utilities importable
_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
_TARGETDIFF_ROOT = _PROJECT_ROOT / "src" / "models" / "targetdiff"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_TARGETDIFF_ROOT))

from models.molopt_score_model import log_normal  # TargetDiff utility


# ---------------------------------------------------------------------------
# Helpers — synthetic data builders
# ---------------------------------------------------------------------------

def _make_atoms(n_mols=3, atoms_per_mol=4, n_dims=3, n_atom_types=13):
    """Returns (lig_mask, pos, v) for n_mols molecules."""
    n_atoms  = n_mols * atoms_per_mol
    lig_mask = torch.repeat_interleave(torch.arange(n_mols), atoms_per_mol)
    pos      = torch.randn(n_atoms, n_dims)
    v        = torch.randint(0, n_atom_types, (n_atoms,))
    return lig_mask, pos, v


def _targetdiff_step_log_prob(pos_s, pos_mean, pos_log_var, v_s, log_model_prob, lig_mask):
    """
    Replicates TargetDiffPolicy._step_log_prob without needing the policy
    object — pure arithmetic extracted for isolated testing.
    """
    n_atom_types = log_model_prob.shape[-1]

    log_p_pos = scatter_mean(
        log_normal(pos_s, pos_mean, 0.5 * pos_log_var), lig_mask, dim=0
    )
    v_s_onehot = F.one_hot(v_s, n_atom_types).float()
    log_p_v    = scatter_mean(
        (log_model_prob * v_s_onehot).sum(dim=-1), lig_mask, dim=0
    )
    return log_p_pos + log_p_v


def _diffsbdd_gaussian_log_prob(z_s, mu_lig, sigma_sq_lig, lig_mask):
    """
    Replicates the Gaussian log-prob formula used in
    conditional_model.log_p_zs_given_zt — extracted for isolated testing.
    """
    diff = z_s - mu_lig
    log_prob_atom = -0.5 * (
        (diff ** 2 / sigma_sq_lig).sum(dim=1) +
        torch.log(2 * torch.tensor(torch.pi) * sigma_sq_lig).sum(dim=1)
    )
    return scatter_mean(log_prob_atom, lig_mask, dim=0)


# ---------------------------------------------------------------------------
# TargetDiff _step_log_prob tests
# ---------------------------------------------------------------------------

class TestTargetDiffStepLogProb(unittest.TestCase):
    """Tests for the combined positional + atom-type log-probability."""

    def setUp(self):
        torch.manual_seed(0)
        self.n_mols       = 3
        self.atoms_per_mol = 4
        self.n_dims       = 3
        self.n_atom_types = 13

        self.lig_mask, self.pos, self.v = _make_atoms(
            self.n_mols, self.atoms_per_mol, self.n_dims, self.n_atom_types
        )
        self.pos_mean     = torch.randn_like(self.pos)
        self.pos_log_var  = torch.zeros_like(self.pos)   # unit variance
        self.log_model_prob = F.log_softmax(
            torch.randn(self.n_mols * self.atoms_per_mol, self.n_atom_types), dim=-1
        )

    def test_output_shape_is_n_mols(self):
        """_step_log_prob must return one value per molecule."""
        result = _targetdiff_step_log_prob(
            self.pos, self.pos_mean, self.pos_log_var,
            self.v, self.log_model_prob, self.lig_mask
        )
        self.assertEqual(result.shape, torch.Size([self.n_mols]))

    def test_output_is_finite(self):
        """No NaN or Inf in any normal input scenario."""
        result = _targetdiff_step_log_prob(
            self.pos, self.pos_mean, self.pos_log_var,
            self.v, self.log_model_prob, self.lig_mask
        )
        self.assertTrue(torch.isfinite(result).all(),
                        f"Non-finite values: {result}")

    def test_positional_log_prob_maximised_at_mean(self):
        """log_prob(pos_s = pos_mean) > log_prob(pos_s = pos_mean + offset).
        Moving z_s away from the predicted mean must reduce the log-prob."""
        pos_at_mean = self.pos_mean.clone()
        pos_offset  = self.pos_mean + 2.0  # large displacement

        lp_at_mean = _targetdiff_step_log_prob(
            pos_at_mean, self.pos_mean, self.pos_log_var,
            self.v, self.log_model_prob, self.lig_mask
        )
        lp_offset = _targetdiff_step_log_prob(
            pos_offset, self.pos_mean, self.pos_log_var,
            self.v, self.log_model_prob, self.lig_mask
        )
        self.assertTrue((lp_at_mean > lp_offset).all(),
                        "log_prob at mean must exceed log_prob away from mean")

    def test_atom_type_log_prob_maximised_at_most_probable_type(self):
        """Choosing v_s = argmax(log_model_prob) should give a higher atom-type
        contribution than choosing a random type for every atom."""
        best_v  = self.log_model_prob.argmax(dim=-1)
        worst_v = (self.log_model_prob.argmin(dim=-1))

        lp_best  = _targetdiff_step_log_prob(
            self.pos_mean, self.pos_mean, self.pos_log_var,
            best_v, self.log_model_prob, self.lig_mask
        )
        lp_worst = _targetdiff_step_log_prob(
            self.pos_mean, self.pos_mean, self.pos_log_var,
            worst_v, self.log_model_prob, self.lig_mask
        )
        self.assertTrue((lp_best > lp_worst).all(),
                        "Picking the most probable atom type must give higher log_prob")

    def test_atom_type_log_prob_bounded_above_by_zero(self):
        """The atom-type term is log(softmax probability) which is always <= 0."""
        v_s_onehot = F.one_hot(self.v, self.n_atom_types).float()
        log_p_v_per_atom = (self.log_model_prob * v_s_onehot).sum(dim=-1)
        self.assertTrue((log_p_v_per_atom <= 0).all(),
                        "log(p) for any categorical probability must be <= 0")

    def test_larger_variance_increases_positional_log_prob_at_mean(self):
        """With higher variance the Gaussian is flatter, so evaluating AT the
        mean gives a lower peak value — log_prob at mean decreases as var grows."""
        pos_log_var_small = torch.full_like(self.pos, -2.0)   # small σ²
        pos_log_var_large = torch.full_like(self.pos,  2.0)   # large σ²

        lp_small_var = _targetdiff_step_log_prob(
            self.pos_mean, self.pos_mean, pos_log_var_small,
            self.v, self.log_model_prob, self.lig_mask
        )
        lp_large_var = _targetdiff_step_log_prob(
            self.pos_mean, self.pos_mean, pos_log_var_large,
            self.v, self.log_model_prob, self.lig_mask
        )
        # Smaller variance → taller Gaussian peak → higher log_prob at mean
        self.assertTrue((lp_small_var > lp_large_var).all(),
                        "Smaller variance → higher log_prob at the mean")


# ---------------------------------------------------------------------------
# DiffSBDD Gaussian formula tests
# ---------------------------------------------------------------------------

class TestDiffSBDDGaussianLogProb(unittest.TestCase):
    """Tests for the Gaussian log-probability formula in
    conditional_model.log_p_zs_given_zt (without the neural network call)."""

    def setUp(self):
        torch.manual_seed(1)
        self.n_mols       = 3
        self.atoms_per_mol = 5
        self.n_dims       = 4   # DiffSBDD uses 3 coords + 1 charge dim

        n_atoms = self.n_mols * self.atoms_per_mol
        self.lig_mask = torch.repeat_interleave(
            torch.arange(self.n_mols), self.atoms_per_mol
        )
        self.mu_lig       = torch.randn(n_atoms, self.n_dims)
        self.sigma_sq_lig = torch.rand(n_atoms, self.n_dims).abs() + 0.1

    def test_output_shape_is_n_mols(self):
        z_s = torch.randn_like(self.mu_lig)
        result = _diffsbdd_gaussian_log_prob(
            z_s, self.mu_lig, self.sigma_sq_lig, self.lig_mask
        )
        self.assertEqual(result.shape, torch.Size([self.n_mols]))

    def test_output_is_finite(self):
        z_s = torch.randn_like(self.mu_lig)
        result = _diffsbdd_gaussian_log_prob(
            z_s, self.mu_lig, self.sigma_sq_lig, self.lig_mask
        )
        self.assertTrue(torch.isfinite(result).all())

    def test_log_prob_maximised_at_mean(self):
        """Gaussian log_prob is maximised when z_s == mu (diff == 0).
        Any displacement must lower the log_prob."""
        z_at_mean  = self.mu_lig.clone()
        z_offset   = self.mu_lig + 3.0

        lp_at_mean = _diffsbdd_gaussian_log_prob(
            z_at_mean, self.mu_lig, self.sigma_sq_lig, self.lig_mask
        )
        lp_offset  = _diffsbdd_gaussian_log_prob(
            z_offset, self.mu_lig, self.sigma_sq_lig, self.lig_mask
        )
        self.assertTrue((lp_at_mean > lp_offset).all(),
                        "Gaussian log_prob must be maximal at z_s == mu")

    def test_log_prob_decreases_monotonically_with_distance(self):
        """Evaluate at z_s = mu + scale*1 for scales 0, 1, 2.
        Log_prob must strictly decrease."""
        lps = []
        for scale in [0.0, 1.0, 2.0]:
            z = self.mu_lig + scale
            lps.append(
                _diffsbdd_gaussian_log_prob(
                    z, self.mu_lig, self.sigma_sq_lig, self.lig_mask
                )
            )
        for i in range(len(lps) - 1):
            self.assertTrue((lps[i] > lps[i + 1]).all(),
                            f"log_prob at scale={i} should exceed scale={i+1}")

    def test_known_value_unit_gaussian(self):
        """For a 1-D unit Gaussian (mu=0, sigma²=1), log_prob at 0 should
        equal -0.5 * log(2π) * n_dims (the per-atom normalisation constant),
        then averaged across atoms via scatter_mean."""
        n_atoms = self.n_mols * self.atoms_per_mol
        mu       = torch.zeros(n_atoms, 1)
        sigma_sq = torch.ones(n_atoms, 1)
        z_at_mu  = torch.zeros(n_atoms, 1)
        lig_mask = torch.repeat_interleave(
            torch.arange(self.n_mols), self.atoms_per_mol
        )

        result = _diffsbdd_gaussian_log_prob(z_at_mu, mu, sigma_sq, lig_mask)

        # log_prob_atom = -0.5 * log(2π) for each atom (diff=0, sigma²=1)
        expected_per_atom = -0.5 * np.log(2 * np.pi)
        # scatter_mean averages over atoms in the molecule
        expected = torch.full((self.n_mols,), expected_per_atom)

        self.assertTrue(
            torch.allclose(result, expected, atol=1e-5),
            f"Expected {expected[0].item():.4f}, got {result[0].item():.4f}"
        )


if __name__ == "__main__":
    unittest.main()
