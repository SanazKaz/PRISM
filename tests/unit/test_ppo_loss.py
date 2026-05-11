"""
Unit tests for src/prism/ppo_tuner/loss.py

Strategy: patch _get_log_probs so we control new_log_probs exactly.
This lets us test the PPO arithmetic in complete isolation from the
policy network and mask-reindexing logic.
"""

import os
os.environ["DEBUG_PPO"] = "0"

import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.prism.ppo_tuner.loss import compute_ppo_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(clip_range=0.2, entropy_coef=0.0, kl_coef=0.0, total_timesteps=1000):
    ppo = SimpleNamespace(clip_range=clip_range, entropy_coef=entropy_coef, kl_coef=kl_coef)
    model = SimpleNamespace(total_timesteps=total_timesteps)
    return SimpleNamespace(ppo=ppo, model=model)


def _make_minibatch(n_mols=4, atoms_per_mol=3, n_features=7,
                    pocket_atoms_per_mol=2, n_timesteps=3,
                    old_log_prob_value=-2.0, advantages=None):
    """Minimal minibatch that compute_ppo_loss can consume."""
    n_atoms = n_mols * atoms_per_mol
    n_pocket = n_mols * pocket_atoms_per_mol
    lig_mask = torch.repeat_interleave(torch.arange(n_mols), atoms_per_mol)
    poc_mask = torch.repeat_interleave(torch.arange(n_mols), pocket_atoms_per_mol)
    if advantages is None:
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    return {
        "latents":      torch.randn(n_atoms,  n_timesteps, n_features),
        "next_latents": torch.randn(n_atoms,  n_timesteps, n_features),
        "old_log_probs": torch.full((n_mols, n_timesteps), old_log_prob_value),
        "molecules": (
            torch.randn(n_atoms,  n_features),
            torch.randn(n_pocket, n_features),
        ),
        "masks": (lig_mask, poc_mask),
        "timesteps": torch.randint(1, 1000, (n_mols, n_timesteps)),
        "advantages": advantages,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPPOLossOutputShape(unittest.TestCase):

    def test_returns_four_scalars(self):
        """compute_ppo_loss must return (loss, approx_kl, clipfrac, entropy)
        each as a 0-d tensor."""
        mb = _make_minibatch()
        cfg = _make_config()

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=torch.full((4,), -2.0)):
            loss, kl, clipfrac, entropy = compute_ppo_loss(None, mb, 0, cfg)

        for name, val in [("loss", loss), ("kl", kl), ("clipfrac", clipfrac), ("entropy", entropy)]:
            self.assertEqual(val.shape, torch.Size([]), f"{name} should be scalar")


class TestRatioEqualsOne(unittest.TestCase):
    """When new_log_probs == old_log_probs, ratio == 1 everywhere.
    Clipping cannot fire, so loss = -mean(advantages) - entropy_coef*entropy."""

    def test_loss_equals_negative_mean_advantage_no_entropy(self):
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mb = _make_minibatch(advantages=advantages, old_log_prob_value=-2.0)
        cfg = _make_config(clip_range=0.2, entropy_coef=0.0, kl_coef=0.0)

        new_log_probs = torch.full((4,), -2.0)  # same as old → ratio = 1

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            loss, _, _, _ = compute_ppo_loss(None, mb, 0, cfg)

        expected = -advantages.mean()
        self.assertTrue(torch.isclose(loss, expected, atol=1e-5),
                        f"Expected {expected.item():.4f}, got {loss.item():.4f}")

    def test_entropy_coef_reduces_loss(self):
        """entropy_coef > 0 subtracts entropy from loss, making it smaller."""
        mb = _make_minibatch(old_log_prob_value=-2.0)
        cfg_no_ent = _make_config(entropy_coef=0.0)
        cfg_ent    = _make_config(entropy_coef=0.1)
        new_log_probs = torch.full((4,), -2.0)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            loss_no_ent, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_no_ent)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            loss_ent, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_ent)

        # entropy = -mean(new_log_probs) = 2.0 > 0, so entropy_coef*entropy > 0
        # loss = policy_loss - entropy_coef*entropy  →  loss_ent < loss_no_ent
        self.assertLess(loss_ent.item(), loss_no_ent.item())


class TestClipping(unittest.TestCase):

    def test_clipping_fires_on_large_ratio(self):
        """When new_log_prob >> old_log_prob, ratio is large and clipping
        should limit the gradient update.  clipfrac should be > 0."""
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])  # all positive
        mb = _make_minibatch(advantages=advantages, old_log_prob_value=-5.0)
        cfg = _make_config(clip_range=0.2, entropy_coef=0.0)

        # new >> old  →  ratio = exp(0 - (-5)) = exp(5) ≈ 148, way outside [0.8, 1.2]
        new_log_probs = torch.zeros(4)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            _, _, clipfrac, _ = compute_ppo_loss(None, mb, 0, cfg)

        self.assertGreater(clipfrac.item(), 0.0,
                           "clipfrac should be >0 when ratio is far outside [1-ε, 1+ε]")

    def test_no_clipping_when_ratio_is_one(self):
        """Ratio == 1 is exactly at the boundary — clipfrac must be 0."""
        mb = _make_minibatch(old_log_prob_value=-2.0)
        cfg = _make_config(clip_range=0.2)
        new_log_probs = torch.full((4,), -2.0)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            _, _, clipfrac, _ = compute_ppo_loss(None, mb, 0, cfg)

        self.assertEqual(clipfrac.item(), 0.0)


class TestKLPenalty(unittest.TestCase):

    def test_kl_coef_zero_does_not_change_loss(self):
        """Setting kl_coef=0 must produce identical loss to having no KL term."""
        mb = _make_minibatch(old_log_prob_value=-2.0)
        new_log_probs = torch.full((4,), -3.0)  # different from old → KL ≠ 0

        cfg_no_kl  = _make_config(kl_coef=0.0,  entropy_coef=0.0)
        cfg_has_kl = _make_config(kl_coef=0.01, entropy_coef=0.0)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs.clone()):
            loss_no_kl, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_no_kl)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs.clone()):
            loss_kl, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_has_kl)

        # KL = (old - new).mean() = (-2 - (-3)) = 1.0
        # loss_kl should be loss_no_kl + 0.01 * 1.0
        expected_kl_penalty = 0.01 * ((-2.0) - (-3.0))
        self.assertTrue(
            torch.isclose(loss_kl - loss_no_kl,
                          torch.tensor(expected_kl_penalty), atol=1e-5),
            f"KL penalty mismatch: got {(loss_kl - loss_no_kl).item():.4f}, "
            f"expected {expected_kl_penalty:.4f}"
        )

    def test_kl_coef_positive_increases_loss_when_policy_drifts(self):
        """When new != old and kl_coef > 0, loss should be higher than with kl_coef=0."""
        mb = _make_minibatch(old_log_prob_value=-2.0)
        # Policy has drifted: new_log_prob < old_log_prob → KL > 0
        new_log_probs = torch.full((4,), -4.0)

        cfg_no_kl = _make_config(kl_coef=0.0,  entropy_coef=0.0)
        cfg_kl    = _make_config(kl_coef=0.1,  entropy_coef=0.0)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs.clone()):
            loss_base, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_no_kl)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs.clone()):
            loss_penalised, _, _, _ = compute_ppo_loss(None, mb, 0, cfg_kl)

        self.assertGreater(loss_penalised.item(), loss_base.item())

    def test_approx_kl_value_is_correct(self):
        """approx_kl returned == mean(old_log_probs - new_log_probs)."""
        old_val, new_val = -2.0, -3.5
        mb = _make_minibatch(old_log_prob_value=old_val)
        cfg = _make_config()

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=torch.full((4,), new_val)):
            _, approx_kl, _, _ = compute_ppo_loss(None, mb, 0, cfg)

        expected = old_val - new_val
        self.assertTrue(torch.isclose(approx_kl, torch.tensor(expected), atol=1e-5))


class TestLossGradients(unittest.TestCase):

    def test_loss_has_gradient(self):
        """policy_loss.backward() must not raise — it needs a grad_fn."""
        mb = _make_minibatch(old_log_prob_value=-2.0)
        cfg = _make_config(kl_coef=0.01)
        # Make new_log_probs a leaf with requires_grad so the graph connects
        new_log_probs = torch.full((4,), -2.5, requires_grad=True)

        with patch("src.prism.ppo_tuner.loss._get_log_probs",
                   return_value=new_log_probs):
            loss, _, _, _ = compute_ppo_loss(None, mb, 0, cfg)

        self.assertIsNotNone(loss.grad_fn, "loss should have a grad_fn")
        loss.backward()  # must not raise


if __name__ == "__main__":
    unittest.main()
