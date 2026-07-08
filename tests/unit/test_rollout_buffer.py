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
from unittest.mock import patch
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


# ---------------------------------------------------------------------------
# Tests: compute_advantages — distributed path (mocked dist)
# ---------------------------------------------------------------------------

class TestComputeAdvantagesDistributed(unittest.TestCase):
    """
    Verify that compute_advantages calls dist.all_reduce when
    dist.is_initialized() returns True, and produces globally-consistent
    normalisation across ranks.

    We mock dist so no real process group is needed.  The mock simulates two
    ranks each holding half the rewards; all_reduce SUM is emulated by
    filling each tensor with the pre-computed global aggregate.
    """

    def _make_buf(self, rewards):
        cfg = _make_config(batch_size=2)
        buf = RolloutBuffer(cfg)
        buf.load_rollout_data(_make_rollout_data(n_mols=len(rewards), rewards=rewards))
        return buf

    def _side_effect_for(self, rank0_rewards, rank1_rewards):
        """Return a side_effect for all_reduce that fills tensors with global aggregates."""
        all_r = torch.cat([rank0_rewards, rank1_rewards])
        expected = [all_r.sum(), (all_r ** 2).sum(), torch.tensor(float(len(all_r)))]
        call_count = [0]

        def _se(tensor, op):
            tensor.fill_(expected[call_count[0]].item())
            call_count[0] += 1

        return _se

    def test_all_reduce_called_when_distributed(self):
        """all_reduce must be invoked exactly 3 times (sum, sum_sq, count)."""
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.9])
        buf = self._make_buf(rewards)

        with patch("src.prism.ppo_tuner.rollout_buffer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.ReduceOp.SUM = "SUM"
            mock_dist.all_reduce.side_effect = self._side_effect_for(rewards, rewards)

            buf.compute_advantages()

        self.assertEqual(mock_dist.all_reduce.call_count, 3)

    def test_global_normalisation_matches_combined(self):
        """Advantages produced under mocked DDP must match single-GPU result
        computed on the full combined reward set."""
        r0 = torch.tensor([0.2, 0.4])
        r1 = torch.tensor([0.6, 0.8])
        all_r = torch.cat([r0, r1])
        expected_mean = all_r.mean()
        expected_std  = (((all_r ** 2).mean() - expected_mean ** 2).clamp(min=0.0) + 1e-8).sqrt()
        expected_adv  = ((r0 - expected_mean) / expected_std).clamp(-3, 3)

        buf = self._make_buf(r0)

        with patch("src.prism.ppo_tuner.rollout_buffer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.ReduceOp.SUM = "SUM"
            mock_dist.all_reduce.side_effect = self._side_effect_for(r0, r1)

            buf.compute_advantages()

        self.assertTrue(
            torch.allclose(buf.advantages, expected_adv, atol=1e-4),
            f"Expected {expected_adv}, got {buf.advantages}",
        )

    def test_single_gpu_fallback_no_all_reduce(self):
        """When dist.is_initialized() is False, all_reduce must NOT be called."""
        rewards = torch.tensor([0.1, 0.5, 0.3, 0.9])
        buf = self._make_buf(rewards)

        with patch("src.prism.ppo_tuner.rollout_buffer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False

            buf.compute_advantages()

        mock_dist.all_reduce.assert_not_called()
        self.assertFalse(buf.advantages.isnan().any())

    def test_distributed_advantages_clamped(self):
        """Clamping to [-3, 3] still applies on the distributed path."""
        r0 = torch.tensor([0.0, 0.0])
        r1 = torch.tensor([0.0, 1e6])
        buf = self._make_buf(r0)

        with patch("src.prism.ppo_tuner.rollout_buffer.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.ReduceOp.SUM = "SUM"
            mock_dist.all_reduce.side_effect = self._side_effect_for(r0, r1)

            buf.compute_advantages()

        self.assertLessEqual(buf.advantages.max().item(),  3.0 + 1e-6)
        self.assertGreaterEqual(buf.advantages.min().item(), -3.0 - 1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_advantages — GRPO (per-pocket) path
# ---------------------------------------------------------------------------

def _make_grpo_config(batch_size=2):
    return SimpleNamespace(ppo=SimpleNamespace(batch_size=batch_size,
                                               use_grpo_advantages=True))


class TestGRPOAdvantages(unittest.TestCase):
    """Per-pocket (within-group) advantage normalisation."""

    def _load(self, rewards, pocket_indices, grpo=True):
        cfg = _make_grpo_config() if grpo else _make_config()
        buf = RolloutBuffer(cfg)
        data = _make_rollout_data(n_mols=len(rewards), rewards=rewards)
        data["pocket_indices"] = pocket_indices
        buf.load_rollout_data(data)
        buf.compute_advantages()
        return buf

    def test_each_group_is_zero_mean(self):
        """Each pocket's advantages should be centred independently."""
        rewards = torch.tensor([0.1, 0.2, 0.3, 5.0, 7.0, 9.0])
        pockets = torch.tensor([0, 0, 0, 1, 1, 1])
        buf = self._load(rewards, pockets)
        for pid in (0, 1):
            grp = buf.advantages[pockets == pid]
            self.assertAlmostEqual(grp.mean().item(), 0.0, places=4)

    def test_differs_from_global_when_pocket_scales_differ(self):
        """GRPO removes pocket-difficulty confound: the high-reward pocket
        gets all-positive advantages under global norm, but is centred (has
        negatives) under GRPO."""
        rewards = torch.tensor([0.1, 0.2, 0.3, 5.0, 7.0, 9.0])
        pockets = torch.tensor([0, 0, 0, 1, 1, 1])
        grpo = self._load(rewards, pockets, grpo=True).advantages
        glob = self._load(rewards, pockets, grpo=False).advantages
        self.assertFalse(torch.allclose(grpo, glob, atol=1e-3))
        self.assertTrue((glob[pockets == 1] > 0).all())   # global: easy pocket all +
        self.assertTrue((grpo[pockets == 1] < 0).any())   # GRPO: centred -> has -

    def test_singleton_group_zero_no_nan(self):
        """A pocket with a single sample has no within-group signal -> 0."""
        rewards = torch.tensor([0.1, 0.9, 0.5])
        pockets = torch.tensor([0, 0, 1])
        buf = self._load(rewards, pockets)
        self.assertFalse(buf.advantages.isnan().any())
        self.assertAlmostEqual(buf.advantages[pockets == 1].item(), 0.0, places=6)

    def test_zero_variance_group_zero_no_nan(self):
        """All-equal rewards within a pocket -> 0 advantages, no divide-by-zero."""
        rewards = torch.tensor([0.7, 0.7, 0.7, 0.2, 0.9])
        pockets = torch.tensor([0, 0, 0, 1, 1])
        buf = self._load(rewards, pockets)
        self.assertFalse(buf.advantages.isnan().any())
        self.assertTrue(torch.allclose(buf.advantages[pockets == 0],
                                       torch.zeros(3), atol=1e-6))

    def test_toggle_off_matches_global(self):
        """Default (flag off) must reproduce the existing global-norm result."""
        rewards = torch.tensor([0.1, 0.2, 0.3, 5.0, 7.0, 9.0])
        pockets = torch.tensor([0, 0, 0, 1, 1, 1])
        off = self._load(rewards, pockets, grpo=False).advantages
        mean, std = rewards.mean(), rewards.std()
        expected = ((rewards - mean) / (std + 1e-8)).clamp(-3, 3)
        self.assertTrue(torch.allclose(off, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
