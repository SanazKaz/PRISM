#!/usr/bin/env python3
"""
Confirm the old_log_probs <-> transition alignment for the PPO timestep windows.

Pure Python — no torch/numpy. Runs in <1s:  python scripts/confirm_window_alignment.py

GROUND TRUTH (src/prism/models/targetdiff_policy.py:170-195)
------------------------------------------------------------
The sampler runs reverse step i = T-1 .. 0 and, each step, appends the NEW state
AND the log-prob that PRODUCED it:

    z_states[k]      = state produced at the k-th append
    old_log_probs[k] = log p( z_states[k] | z_states[k-1] )

The buffer then sets (rollout_collector.py:281-282):

    latents      = z_states[:, :-1]        # z_states[0 .. S-2]
    next_latents = z_states[:, 1:]         # z_states[1 .. S-1]

so transition j is ( latents[j]=z_states[j]  ->  next_latents[j]=z_states[j+1] ),
and its sampling log-prob is the one that PRODUCED z_states[j+1], i.e.

        transition j   <->   old_log_probs[j+1]

Consequences:
  * old_log_probs has S entries; there are only S-1 transitions.
  * the unmatched entry is old_log_probs[0]  (the z_init -> z_states[0] step).
  * the aligned slice is old_log_probs[-T:]  (drop the FRONT entry).

This script tags old_log_probs[k] = k so every kept entry is identifiable, builds
the three windows EXACTLY as ppo_algorithm.py does, and checks which slice
recovers the correct per-transition entry (j+1) for every kept transition j.

Expected result (current / un-fixed code):
    'last'  -> correctly aligned
    'first' -> OFF BY ONE
    'band'  -> OFF BY ONE
and applying old_log_probs[-T:] before slicing fixes 'first'/'band' while leaving
'last' byte-identical.
"""

S = 8                       # stored states  => len(old_log_probs) == S
T = S - 1                   # transitions    => len(latents/next_latents/timesteps)
total_timesteps = S

# tag[k] = k  ("the log-prob that produced state k")
old_log_probs = list(range(S))
# the collector's timestep grid: arange(total-1, -1, -1)[:T]
timesteps = list(range(total_timesteps - 1, -1, -1))[:T]
# the correct old_log_probs index for transition j is j+1
true_for = [j + 1 for j in range(T)]


def report(name, kept_idx, picked):
    want = [true_for[j] for j in kept_idx]
    ok = picked == want
    print(f"  {name:5s} | transitions kept: {kept_idx}")
    print(f"        | correct old idx : {want}")
    print(f"        | slice picked    : {picked}   -> {'ALIGNED' if ok else 'OFF BY ONE'}")
    return ok


print(f"states S={S}, transitions T={T}, total_timesteps={total_timesteps}")
print(f"old_log_probs (tagged) = {old_log_probs}")
print(f"timesteps grid         = {timesteps}\n")

# ---- windows exactly as ppo_algorithm.py:148-173 (CURRENT, un-fixed) ----------
k = 3

# 'last'  : every tensor sliced [:, -k:]
last_idx = list(range(T - k, T))          # transitions covered by latents[-k:]
last_pick = old_log_probs[S - k:]         # old_log_probs[-k:]

# 'first' : every tensor sliced [:, :k]
first_idx = list(range(0, k))
first_pick = old_log_probs[:k]

# 'band'  : keep = {j : t_lo <= timesteps[j] <= t_hi}; old_log_probs[:, keep]
t_lo, t_hi = 2, 4
band_idx = [j for j in range(T) if t_lo <= timesteps[j] <= t_hi]
band_pick = [old_log_probs[j] for j in band_idx]

print("=== CURRENT (un-fixed) slicing ===")
r_last = report("last", last_idx, last_pick)
r_first = report("first", first_idx, first_pick)
r_band = report("band", band_idx, band_pick)

# ---- the proposed fix: align old_log_probs[-T:] BEFORE slicing -----------------
old_aligned = old_log_probs[-T:]          # drop the unmatched front entry
fix_band_pick = [old_aligned[j] for j in band_idx]
fix_first_pick = old_aligned[:k]
# 'last' after fix: old_aligned[-k:] — show it equals the un-fixed 'last'
fix_last_pick = old_aligned[T - k:]

print("\n=== WITH FIX (old_log_probs[-T:], then slice) ===")
report("first", first_idx, fix_first_pick)
report("band", band_idx, fix_band_pick)
print(f"  last  | fix gives {fix_last_pick} ; un-fixed gave {last_pick} "
      f"-> {'IDENTICAL' if fix_last_pick == last_pick else 'CHANGED'}")

print()
assert r_last, "FAIL: expected 'last' to be aligned on current code"
assert not r_first, "FAIL: expected 'first' to be OFF BY ONE on current code"
assert not r_band, "FAIL: expected 'band' to be OFF BY ONE on current code"
assert fix_band_pick == [true_for[j] for j in band_idx], "FAIL: fix did not align 'band'"
assert fix_last_pick == last_pick, "FAIL: fix changed 'last' (it must not)"
print("CONFIRMED:")
print("  - 'last'  is correctly aligned (and the [-T:] fix leaves it identical)")
print("  - 'first' and 'band' are OFF BY ONE: they pair transition j with")
print("    old_log_probs[j] instead of old_log_probs[j+1]")
print("  - aligning old_log_probs[-T:] before slicing fixes them")
