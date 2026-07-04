"""
Regression test for apply_timestep_window() — old_log_probs <-> transition alignment.

Background
----------
The sampler (targetdiff_policy.sample_given_pocket) appends, each reverse step, the
NEW state and the log-prob that produced it, so

    old_log_probs[k] = log p( z_states[k] | z_states[k-1] )      (S entries)

The buffer sets latents = z_states[:, :-1], next_latents = z_states[:, 1:], so
transition j = (z_states[j] -> z_states[j+1]) and its sampling log-prob is
old_log_probs[j+1]  (S-1 transitions; the unmatched entry is old_log_probs[0]).

Hence the aligned slice is old_log_probs[-T:]. apply_timestep_window() must align
old_log_probs FIRST, then slice — otherwise 'first'/'band' pair transition j with
old_log_probs[j] (off by one). This test pins that invariant.

We tag old_log_probs[:, k] = k and latents[:, j, :] = j so the correct post-window
pairing is: for every kept position, old_log_probs == latents_tag + 1.
"""

import torch

from src.prism.ppo_tuner.timestep_window import apply_timestep_window


S = 8                     # stored states  -> old_log_probs has S columns
T = S - 1                 # transitions    -> latents/next_latents/timesteps have T columns
TOTAL = S                 # total_timesteps used to build the timestep grid
M = 2                     # molecules
N = 3                     # ligand atoms
D = 2                     # latent feature dim


def _make_rollout():
    """Tagged rollout: old_log_probs[:, k] = k ; latents[:, j, :] = j ;
    timesteps row = arange(TOTAL-1, -1, -1)[:T] (the collector's grid)."""
    old_log_probs = torch.arange(S).float().unsqueeze(0).repeat(M, 1)          # [M, S]
    latents = torch.arange(T).float().view(1, T, 1).repeat(N, 1, D)            # [N, T, D]
    next_latents = latents.clone()
    timesteps = torch.arange(TOTAL - 1, -1, -1)[:T].unsqueeze(0).repeat(M, 1)  # [M, T]
    return {
        "old_log_probs": old_log_probs,
        "latents": latents,
        "next_latents": next_latents,
        "timesteps": timesteps,
    }


def _kept_transition_tags(rd):
    """The transition index tag kept in each column after windowing (from latents)."""
    return rd["latents"][0, :, 0].long().tolist()


def test_last_window_is_aligned_and_unchanged():
    rd = _make_rollout()
    k = 3
    current_k = apply_timestep_window(rd, window="last", train_timesteps=k)
    assert current_k == k
    kept = _kept_transition_tags(rd)                 # last 3 transitions
    assert kept == [4, 5, 6]
    # old_log_probs[j] must equal transition_tag + 1 for every kept column
    for col, tag in enumerate(kept):
        assert torch.all(rd["old_log_probs"][:, col] == tag + 1)
    assert rd["old_log_probs"][0].tolist() == [5.0, 6.0, 7.0]
    assert rd["timesteps"][0].tolist() == [3, 2, 1]  # 7 - tag


def test_first_window_is_aligned_after_fix():
    rd = _make_rollout()
    k = 3
    current_k = apply_timestep_window(rd, window="first", train_timesteps=k)
    assert current_k == k
    kept = _kept_transition_tags(rd)                 # first 3 transitions
    assert kept == [0, 1, 2]
    for col, tag in enumerate(kept):
        assert torch.all(rd["old_log_probs"][:, col] == tag + 1)
    assert rd["old_log_probs"][0].tolist() == [1.0, 2.0, 3.0]


def test_band_window_is_aligned_after_fix():
    rd = _make_rollout()
    current_k = apply_timestep_window(rd, window="band", train_timesteps=999, t_lo=2, t_hi=4)
    kept = _kept_transition_tags(rd)
    assert kept == [3, 4, 5]                          # timesteps 4,3,2 fall in [2,4]
    assert current_k == 3
    for col, tag in enumerate(kept):
        assert torch.all(rd["old_log_probs"][:, col] == tag + 1)
    assert rd["old_log_probs"][0].tolist() == [4.0, 5.0, 6.0]
    assert rd["timesteps"][0].tolist() == [4, 3, 2]


def test_band_documents_the_old_off_by_one():
    """Without the [-T:] alignment, front-indexing would pick old_log_probs[j]
    (== tag) instead of the correct tag+1. This documents the bug the fix closes."""
    rd = _make_rollout()
    t_row = rd["timesteps"][0]
    keep = ((t_row >= 2) & (t_row <= 4)).nonzero(as_tuple=True)[0]
    naive_front = rd["old_log_probs"][:, keep]        # the OLD (buggy) behaviour
    assert naive_front[0].tolist() == [3.0, 4.0, 5.0]  # == tag, i.e. off by one (should be tag+1)


def test_all_window_tensors_sliced_consistently():
    rd = _make_rollout()
    apply_timestep_window(rd, window="band", train_timesteps=999, t_lo=2, t_hi=4)
    L = rd["timesteps"].shape[1]
    assert rd["latents"].shape[1] == L
    assert rd["next_latents"].shape[1] == L
    assert rd["old_log_probs"].shape[1] == L
    # latents and next_latents kept the same transition columns
    assert torch.equal(rd["latents"][0, :, 0], rd["next_latents"][0, :, 0])


def test_band_requires_bounds():
    rd = _make_rollout()
    try:
        apply_timestep_window(rd, window="band", train_timesteps=999)
    except ValueError:
        return
    raise AssertionError("band window without t_lo/t_hi should raise ValueError")


def test_empty_band_raises():
    rd = _make_rollout()
    try:
        apply_timestep_window(rd, window="band", train_timesteps=999, t_lo=900, t_hi=999)
    except ValueError:
        return
    raise AssertionError("band selecting 0 steps should raise ValueError")
