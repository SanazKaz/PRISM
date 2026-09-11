"""
Tests for the frozen-reference-policy KL anchor.

Two things are pinned:

1. RolloutCollector._apply_ref_kl_penalty (src/prism/ppo_tuner/rollout_collector.py)
   — the actual subtraction R' = R - beta * KL(pi_old || pi_ref), and the
   old_log_probs <-> latents alignment it depends on (old_log_probs has one
   more column than latents/timesteps; the aligned slice is old_log_probs[-T:],
   same convention as test_timestep_window_alignment.py). This code path never
   branches on model_type, so one test covers both backbones.

2. PPOFineTuner.__init__ (src/prism/ppo_tuner/lightning_module.py) — that a
   frozen reference policy is built for BOTH diffsbdd and targetdiff when
   ref_kl_coef > 0, and for neither when it is 0. Before this change, the
   ref_policy branch existed only for targetdiff, so ref_kl_coef > 0 silently
   did nothing for diffsbdd runs.
"""

import os
os.environ["DEBUG_PPO"] = "0"

import sys
from pathlib import Path

# ppo_algorithm.py (imported transitively via lightning_module in Part 2)
# does `from utils import permute_timesteps`, the vendored DiffSBDD flat
# module — needs this on sys.path, same as tests/unit/test_encoding_decoding.py.
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_DIFFSBDD_ROOT = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_DIFFSBDD_ROOT))

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from src.prism.ppo_tuner.rollout_collector import RolloutCollector


# ---------------------------------------------------------------------------
# Part 1 — the KL subtraction itself
# ---------------------------------------------------------------------------

def _make_config(ref_kl_coef=0.1):
    # model.total_timesteps is read by _get_log_probs (mocked below in most
    # tests, but the real _apply_ref_kl_penalty still reaches for it before
    # calling in).
    return SimpleNamespace(
        ppo=SimpleNamespace(ref_kl_coef=ref_kl_coef),
        model=SimpleNamespace(total_timesteps=1000),
    )


def _make_collector(ref_kl_coef=0.1):
    """A RolloutCollector needs policy_network only to read .device; the KL
    penalty itself calls ref_policy through the mocked _get_log_probs below,
    so a real diffusion model is never required."""
    dummy_policy = nn.Linear(2, 2)  # any nn.Module with parameters
    dummy_ref = nn.Linear(2, 2)
    config = _make_config(ref_kl_coef=ref_kl_coef)
    collector = RolloutCollector(
        policy_network=dummy_policy,
        reward_function=None,
        config=config,
        ref_policy=dummy_ref,
    )
    return collector


class TestApplyRefKlPenalty(unittest.TestCase):

    def test_subtracts_beta_times_mean_kl(self):
        """R' = R - beta * mean_t(old_log_probs - ref_log_probs), per molecule."""
        M, T = 3, 4  # molecules, transitions
        beta = 0.1
        collector = _make_collector(ref_kl_coef=beta)

        # old_log_probs has T+1 columns (full diffusion trace); latents/
        # next_latents/timesteps have T (one fewer, matching z_states[:,:-1]).
        old_log_probs = torch.arange(M * (T + 1), dtype=torch.float32).reshape(M, T + 1)
        latents = torch.zeros(5, T, 2)       # (num_atoms, T, D) — content unused
        next_latents = torch.zeros(5, T, 2)
        timesteps = torch.zeros(M, T)
        rewards = torch.tensor([1.0, 2.0, 3.0])

        rollout_data = {
            'old_log_probs': old_log_probs,
            'latents': latents,
            'next_latents': next_latents,
            'timesteps': timesteps,
            'molecules': (None, None),
            'masks': (None, None),
            'rewards': rewards,
        }

        # Fixed, known ref log-probs — independent of what the mocked policy sees.
        ref_log_probs = torch.ones(M, T) * 0.5

        with patch(
            "src.prism.ppo_tuner.rollout_collector._get_log_probs",
            return_value=torch.zeros(M),
        ) as mock_get_lp:
            # _get_log_probs is called once per timestep inside the loop and
            # appended into a list, so make it return one ref value per call
            # by cycling through ref_log_probs columns.
            mock_get_lp.side_effect = [ref_log_probs[:, t] for t in range(T)]

            out = collector._apply_ref_kl_penalty(rollout_data)

        old_lp_aligned = old_log_probs[:, -T:]                    # alignment under test
        expected_kl = (old_lp_aligned - ref_log_probs).mean(dim=1)  # (M,)
        expected_rewards = rewards - beta * expected_kl

        torch.testing.assert_close(out['rewards'], expected_rewards)
        # KL against an on-policy log-prob should never be negative on average
        # in this construction (old_lp_aligned grows with each row/column);
        # sanity-check the sign convention rather than the exact values.
        self.assertTrue(torch.all(expected_kl >= 0))

    def test_alignment_uses_last_T_columns_not_first_T(self):
        """Regression guard: swapping -T: for :T must change the result,
        otherwise this test (and the real code) could pass by accident."""
        M, T = 2, 3
        beta = 1.0
        collector = _make_collector(ref_kl_coef=beta)

        old_log_probs = torch.arange(M * (T + 1), dtype=torch.float32).reshape(M, T + 1)
        rollout_data = {
            'old_log_probs': old_log_probs,
            'latents': torch.zeros(4, T, 2),
            'next_latents': torch.zeros(4, T, 2),
            'timesteps': torch.zeros(M, T),
            'molecules': (None, None),
            'masks': (None, None),
            'rewards': torch.zeros(M),
        }

        with patch(
            "src.prism.ppo_tuner.rollout_collector._get_log_probs",
        ) as mock_get_lp:
            mock_get_lp.side_effect = [torch.zeros(M) for _ in range(T)]
            out = collector._apply_ref_kl_penalty(dict(rollout_data))

        last_T_kl = old_log_probs[:, -T:].mean(dim=1)
        first_T_kl = old_log_probs[:, :T].mean(dim=1)
        self.assertFalse(torch.allclose(last_T_kl, first_T_kl))
        torch.testing.assert_close(out['rewards'], -beta * last_T_kl)

    def test_no_penalty_when_ref_policy_absent(self):
        """collect() must skip the penalty entirely when ref_policy is None,
        regardless of ref_kl_coef — this is the flag RolloutCollector.collect
        checks before ever calling _apply_ref_kl_penalty."""
        config = _make_config(ref_kl_coef=0.5)
        collector = RolloutCollector(
            policy_network=nn.Linear(2, 2),
            reward_function=None,
            config=config,
            ref_policy=None,
        )
        self.assertIsNone(collector.ref_policy)
        # Mirrors the guard in collect(): `if self.ref_policy is not None and
        # self._ref_kl_coef > 0.0`.
        should_apply = collector.ref_policy is not None and collector._ref_kl_coef > 0.0
        self.assertFalse(should_apply)


# ---------------------------------------------------------------------------
# Part 2 — PPOFineTuner wires a frozen ref policy for EITHER backbone
# ---------------------------------------------------------------------------

def _make_lightning_config(model_type, ref_kl_coef):
    # Top level must be a real argparse.Namespace (not SimpleNamespace) —
    # Lightning's save_hyperparameters() only accepts Namespace/dict/its own
    # AttributeDict, matching how real configs are built (YAML -> Namespace).
    # Nested blocks can stay as SimpleNamespace; only attribute access on them
    # is ever needed.
    return argparse.Namespace(
        model_type=model_type,
        gpus=0,  # force CPU — no CUDA dependency in this test
        freeze_except=['dummy'],
        datadir='/fake/datadir',
        grad_logging=None,
        model=SimpleNamespace(total_timesteps=500),
        ppo=SimpleNamespace(ref_kl_coef=ref_kl_coef),
    )


def _dummy_policy_factory():
    """Returns a fresh nn.Module each call, so identity checks can confirm
    the ref policy is a separate object from the trainable one."""
    return nn.Linear(2, 2)


class TestRefPolicyWiringBothBackbones(unittest.TestCase):
    """Before this change, build_targetdiff_policy was called twice (policy +
    ref) only inside the `model_type == 'targetdiff'` branch; the `else`
    (diffsbdd) branch had no equivalent, so ref_kl_coef > 0 was silently a
    no-op for diffsbdd. These tests pin that both branches now match."""

    def _build(self, model_type, ref_kl_coef):
        # Guard against cross-test sys.modules pollution: another test file
        # in this suite (test_log_probs.py) puts TargetDiff's utils/ package
        # on sys.path and, through its own imports, can leave sys.modules
        # ['utils'] pointing at it. `import utils` inside ppo_algorithm.py
        # (lazily triggered by importing lightning_module below) checks
        # sys.modules first and ignores sys.path entirely if already cached,
        # so a stale entry here would resolve to the wrong "utils" — DiffSBDD's
        # flat utils.py is what ppo_algorithm.py actually needs. Evict it so
        # this import re-resolves fresh, same fix test_encoding_decoding.py
        # documents needing when both vendored roots are on sys.path at once.
        sys.modules.pop('utils', None)
        if str(_DIFFSBDD_ROOT) in sys.path:
            sys.path.remove(str(_DIFFSBDD_ROOT))
        sys.path.insert(0, str(_DIFFSBDD_ROOT))

        from src.prism.ppo_tuner.lightning_module import PPOFineTuner

        config = _make_lightning_config(model_type, ref_kl_coef)

        with patch(
            "src.prism.ppo_tuner.lightning_module.build_diffsbdd_policy"
        ) as mock_diffsbdd, patch(
            "src.prism.ppo_tuner.lightning_module.build_targetdiff_policy"
        ) as mock_targetdiff, patch(
            "src.prism.ppo_tuner.lightning_module.get_reward_manager",
            return_value=MagicMock(),
        ), patch(
            "src.prism.ppo_tuner.lightning_module.PPOAlgorithm",
            return_value=MagicMock(),
        ), patch(
            "src.prism.models.targetdiff_inference.make_targetdiff_reconstruction_fn",
            return_value=lambda *a, **k: None,
        ):
            mock_diffsbdd.side_effect = lambda **kw: (
                _dummy_policy_factory(), MagicMock(), {'dummy': True}
            )
            mock_targetdiff.side_effect = lambda **kw: (
                _dummy_policy_factory(), {'dummy': True}
            )

            tuner = PPOFineTuner(
                config=config,
                node_histogram=[1, 2, 3],
                warm_start_checkpoint='/fake/ckpt.ckpt',
            )

        builder = mock_diffsbdd if model_type == 'diffsbdd' else mock_targetdiff
        return tuner, builder

    def test_diffsbdd_builds_frozen_ref_policy_when_kl_enabled(self):
        tuner, builder = self._build('diffsbdd', ref_kl_coef=0.05)

        self.assertEqual(builder.call_count, 2, "expected one call for the "
                          "trainable policy and one for the ref policy")
        self.assertIsNotNone(tuner.ref_policy)
        self.assertIsNot(tuner.ref_policy, tuner.policy,
                          "ref policy must be a separate instance, not an alias")
        self.assertFalse(tuner.ref_policy.training, "ref policy must be in eval mode")
        self.assertTrue(
            all(not p.requires_grad for p in tuner.ref_policy.parameters()),
            "ref policy parameters must be frozen",
        )

    def test_targetdiff_builds_frozen_ref_policy_when_kl_enabled(self):
        tuner, builder = self._build('targetdiff', ref_kl_coef=0.05)

        self.assertEqual(builder.call_count, 2)
        self.assertIsNotNone(tuner.ref_policy)
        self.assertIsNot(tuner.ref_policy, tuner.policy)
        self.assertFalse(tuner.ref_policy.training)
        self.assertTrue(all(not p.requires_grad for p in tuner.ref_policy.parameters()))

    def test_diffsbdd_no_ref_policy_when_kl_disabled(self):
        tuner, builder = self._build('diffsbdd', ref_kl_coef=0.0)

        self.assertEqual(builder.call_count, 1, "only the trainable policy should be built")
        self.assertIsNone(tuner.ref_policy)

    def test_targetdiff_no_ref_policy_when_kl_disabled(self):
        tuner, builder = self._build('targetdiff', ref_kl_coef=0.0)

        self.assertEqual(builder.call_count, 1)
        self.assertIsNone(tuner.ref_policy)


if __name__ == '__main__':
    unittest.main()
