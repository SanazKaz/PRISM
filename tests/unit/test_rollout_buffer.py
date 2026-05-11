"""
Unit tests for src/prism/ppo_tuner/rollout_buffer.py

Tests advantage normalisation, degenerate-reward handling, clamping,
and minibatch coverage/consistency — all without a real model or GPU.
"""

import os
os.environ["DEBUG_PPO"] = "0"  # disable global-state tracking in debug utils

import torch
import unittest
from types import SimpleNamespace
from tests.ppo_debug_utils import reset_seen_mb_ids

from src.prism.ppo_tuner.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(batch_size=2):
    return SimpleNamespace(ppo=SimpleNamespace(batch_size=batch_size))


def _make_rollout_data(n_mols=4, atoms_per_mol=3, pocket_atoms_per_mol=2,
                       n_timesteps=5, n_features=4, n_pocket_features=3,
                       rewards=None):
    """Builds a complete rollout_data dict the buffer can load."""
    n_atoms  = n_mols * atoms_per_mol
    n_pocket = n_mols * pocket_atoms_per_mol

    lig_mask  = torch.repeat_interleave(torch.arange(n_mols), atoms_per_mol)
    poc_mask  = torch.repeat_interleave(torch.arange(n_mols), pocket_atoms_per_mol)

    if rewards is None:
        rewards = torch.arange(n_mols, dtype=torch.float32)  # distinct values

    return {
        "rewards":       rewards,
        "raw_score":     rewards.clone(),
        "pocket_indices": torch.zeros(n_mols, dtype=torch.long),
        "molecules":     (
            torch.randn(n_atoms,  n_features),
            torch.randn(n_pocket, n_pocket_features),
        ),
        "masks":         (lig_mask, poc_mask),
        "latents":       torch.randn(n_atoms, n_timesteps, n_features),
        "next_latents":  torch.randn(n_atoms, n_timesteps, n_features),
        "old_log_probs": torch.randn(n_mols, n_timesteps),
        "timesteps":     torch.randint(1, 100, (n_mols, n_timesteps)),
    }


# ---------------------------------------------------------------------------
# Tests: compute_advantages
# ---------------------------------------------------------------------------

class TestComputeAdvantages(unittest.TestCase):

    def setUp(self):
        self.cfg = _make_config()

    def test_advantages_are_normalised(self):
        """With varied rewards, advantages should have mean ≈ 0 and std ≈ 1
        (before clamping, for rewards that don't require clamping)."""
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.9])
        buf = RolloutBuffer(self.cfg)
        buf.load_rollout_data(_make_rollout_data(rewards=rewards))
        buf.compute_advantages()

        self.assertAlmostEqual(buf.advantages.mean().item(), 0.0, places=4,
                               msg="Advantages should have zero mean")
        self.assertAlmostEqual(buf.advantages.std().item(), 1.0, places=2,
                               msg="Advantages should have unit std (if not clamped)")

    def test_identical_rewards_produce_zero_advantages(self):
        """When all rewards are the same, std == 0. The buffer should fall
        back to centring (reward - mean == 0) without dividing by zero."""
        rewards = torch.ones(4) * 0.7
        buf = RolloutBuffer(self.cfg)
        buf.load_rollout_data(_make_rollout_data(rewards=rewards))
        buf.compute_advantages()

        self.assertTrue(torch.allclose(buf.advantages, torch.zeros(4)),
                        "Identical rewards should give zero advantages")

    def test_single_molecule_does_not_crash(self):
        """A batch of size 1 has undefined std — buffer should handle it."""
        rewards = torch.tensor([0.5])
        data = _make_rollout_data(n_mols=1, atoms_per_mol=3,
                                  pocket_atoms_per_mol=2, rewards=rewards)
        buf = RolloutBuffer(self.cfg)
        buf.load_rollout_data(data)
        buf.compute_advantages()  # must not raise or produce NaN

        self.assertFalse(buf.advantages.isnan().any(), "Advantages must not be NaN")

    def test_advantages_clamped_to_three(self):
        """Extreme rewards should be clamped to [-3, 3]."""
        rewards = torch.tensor([0.0, 0.0, 0.0, 1e6])
        buf = RolloutBuffer(self.cfg)
        buf.load_rollout_data(_make_rollout_data(rewards=rewards))
        buf.compute_advantages()

        self.assertLessEqual(buf.advantages.max().item(),  3.0 + 1e-6)
        self.assertGreaterEqual(buf.advantages.min().item(), -3.0 - 1e-6)

    def test_compute_before_load_raises(self):
        buf = RolloutBuffer(self.cfg)
        with self.assertRaises(ValueError):
            buf.compute_advantages()


# ---------------------------------------------------------------------------
# Tests: get_minibatches
# ---------------------------------------------------------------------------

class TestGetMinibatches(unittest.TestCase):

    def setUp(self):
        reset_seen_mb_ids()  # clear global tracking state between tests

    def test_all_molecules_appear_exactly_once(self):
        """Iterating all minibatches must yield every molecule ID exactly once."""
        n_mols = 6
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        data = _make_rollout_data(n_mols=n_mols)
        buf.load_rollout_data(data)
        buf.compute_advantages()

        reset_seen_mb_ids()
        seen_mol_ids = []
        for mb in buf.get_minibatches():
            lig_mask = mb["masks"][0]
            seen_mol_ids.extend(torch.unique(lig_mask).tolist())

        self.assertEqual(sorted(seen_mol_ids), list(range(n_mols)),
                         "Every molecule must appear in exactly one minibatch")

    def test_minibatch_count_is_correct(self):
        """With n_mols=6 and batch_size=2, expect 3 minibatches."""
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        buf.load_rollout_data(_make_rollout_data(n_mols=6))
        buf.compute_advantages()

        reset_seen_mb_ids()
        n_batches = sum(1 for _ in buf.get_minibatches())
        self.assertEqual(n_batches, 3)

    def test_lig_and_pocket_masks_cover_same_molecules(self):
        """In each minibatch, lig_mask and poc_mask must reference the same
        set of molecule IDs."""
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        buf.load_rollout_data(_make_rollout_data(n_mols=4))
        buf.compute_advantages()

        reset_seen_mb_ids()
        for mb in buf.get_minibatches():
            lig_mask, poc_mask = mb["masks"]
            lig_ids = set(torch.unique(lig_mask).tolist())
            poc_ids = set(torch.unique(poc_mask).tolist())
            self.assertEqual(lig_ids, poc_ids,
                             "lig_mask and poc_mask must cover the same molecule IDs")

    def test_advantages_length_matches_molecules_in_minibatch(self):
        """mb['advantages'] must have the same length as the number of
        molecules in the minibatch."""
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        buf.load_rollout_data(_make_rollout_data(n_mols=4))
        buf.compute_advantages()

        reset_seen_mb_ids()
        for mb in buf.get_minibatches():
            n_mol_in_mb = len(torch.unique(mb["masks"][0]))
            self.assertEqual(mb["advantages"].shape[0], n_mol_in_mb)

    def test_latent_atom_count_matches_mask(self):
        """mb['latents'].shape[0] must equal the total atoms in the minibatch."""
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        buf.load_rollout_data(_make_rollout_data(n_mols=4, atoms_per_mol=3))
        buf.compute_advantages()

        reset_seen_mb_ids()
        for mb in buf.get_minibatches():
            n_atoms_in_mb = mb["masks"][0].shape[0]
            self.assertEqual(mb["latents"].shape[0], n_atoms_in_mb)
            self.assertEqual(mb["next_latents"].shape[0], n_atoms_in_mb)

    def test_get_minibatches_before_advantages_raises(self):
        buf = RolloutBuffer(_make_config())
        buf.load_rollout_data(_make_rollout_data())
        # advantages not computed yet
        with self.assertRaises(ValueError):
            list(buf.get_minibatches())

    def test_empty_rollout_sets_data_loaded_false(self):
        buf = RolloutBuffer(_make_config())
        empty = {"rewards": torch.tensor([])}
        buf.load_rollout_data(empty)
        self.assertFalse(buf.data_loaded)


if __name__ == "__main__":
    unittest.main()
