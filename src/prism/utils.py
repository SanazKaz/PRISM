import torch
import numpy as np


# used in training step for timestep permutations

def permute_timesteps(rollout_data, device):
    """
    Randomly permute diffusion timesteps *per molecule* for:
        - molecule-wise tensors  [B, T]
        - atom-wise  tensors     [N, T, F]   (use lig_mask for lookup)

    Adds no CPU sync, no Python loops, O(N+T) memory.
    """
    B, T = rollout_data["timesteps"].shape
    perms = torch.stack([torch.randperm(T, device=device) for _ in range(B)])  # [B, T]

    # ---------- molecule-wise tensors ----------
    for key in ("timesteps", "old_log_probs"):
        if key in rollout_data and rollout_data[key] is not None:
            rollout_data[key] = rollout_data[key].gather(1, perms)

    # ---------- atom-wise tensors --------------
    for key in ("latents", "next_latents"):
        if key not in rollout_data or rollout_data[key] is None:
            continue

        x        = rollout_data[key]                     # [N, T, F]
        lig_mask = rollout_data["masks"][0]              # [N]  global IDs
        N, _, F  = x.shape

        # build lookup: global-ID → local batch idx (0..B-1)
        ids          = torch.unique(lig_mask)
        id2local     = {int(gid): idx for idx, gid in enumerate(ids.tolist())}
        local_idx    = torch.tensor([id2local[int(g)] for g in lig_mask.tolist()],
                                    device=device, dtype=torch.long)          # [N]

        # perms[local_idx] gives a [N, T] index tensor
        gather_idx = perms[local_idx]          # [N, T]

        # expand for feature dim and gather
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, F)   # [N, T, F]
        rollout_data[key] = x.gather(1, gather_idx)

    return rollout_data