"""
Timestep-window selection for PPO rollouts (dependency-free, unit-testable).

Kept in its own module (no torch/utils imports) so it can be imported in tests
without pulling in ppo_algorithm's heavy module-level imports.
"""

_WINDOW_KEYS = ("latents", "next_latents", "old_log_probs", "timesteps")


def apply_timestep_window(rollout_data, window, train_timesteps, t_lo=None, t_hi=None):
    """Slice the per-step rollout tensors to the chosen training window, in place.

    The stored reverse sequence runs [t=T-1 (noisy) ... t=0 (near-clean)] (see
    RolloutCollector: timesteps_1d = arange(T-1, -1, -1)). Windows:
      'last'  (default) — final K low-noise steps (original behaviour).
      'first'           — initial K high-noise steps.
      'band'            — steps whose diffusion timestep t is in [t_lo, t_hi]
                          (timestep units; ignores train_timesteps).

    Alignment: ``old_log_probs`` spans the full chain (T+1 entries, one per reverse
    step including the z_init -> z_{T-1} step), but ``latents``/``next_latents``/
    ``timesteps`` span the T transitions (z_states[:, :-1] / [:, 1:]). The sampler
    appends each new state with the log-prob that produced it, so transition j
    pairs with ``old_log_probs[j+1]`` and the aligned slice is ``old_log_probs[-T:]``
    (drop the unmatched front entry). We align FIRST, then slice — otherwise 'first'
    and 'band' index from the front and pair transition j with old_log_probs[j]
    (off by one), injecting the per-step normalization offset into the PPO ratio and
    corrupting the clip. 'last' is unaffected (its [-k:] slice was already aligned).

    Returns current_k (number of kept timesteps).
    """
    T = rollout_data["timesteps"].shape[1]
    olp = rollout_data.get("old_log_probs")
    if olp is not None and olp.shape[1] != T:
        rollout_data["old_log_probs"] = olp[:, -T:]

    if window == 'band':
        if t_lo is None or t_hi is None:
            raise ValueError("timestep_window='band' requires ppo.t_lo and ppo.t_hi.")
        t_lo, t_hi = int(t_lo), int(t_hi)
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        # Deterministic schedule => every molecule shares the timestep row; row 0
        # is representative (matches the positional-slicing assumption).
        t_row = rollout_data["timesteps"][0]
        keep = ((t_row >= t_lo) & (t_row <= t_hi)).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            raise ValueError(
                f"timestep_window='band' selected 0 steps for t in [{t_lo}, {t_hi}]; "
                f"available t range is [{int(t_row.min())}, {int(t_row.max())}]."
            )
        current_k = int(keep.numel())
        for key in _WINDOW_KEYS:
            if rollout_data.get(key) is not None:
                rollout_data[key] = rollout_data[key][:, keep]
    else:
        current_k = int(train_timesteps)
        for key in _WINDOW_KEYS:
            if rollout_data.get(key) is not None:
                if window == 'first':
                    rollout_data[key] = rollout_data[key][:, :current_k]
                else:
                    rollout_data[key] = rollout_data[key][:, -current_k:]
    return current_k
