# src/prism/ppo_tuner/loss.py

import torch
from tests.ppo_debug_utils import assert_same_ids, dbg_tensor

def compute_ppo_loss(policy_network, minibatch, timestep_idx, config):
    """
    Computes the PPO loss for a single minibatch and a single timestep.

    Args:
        policy_network (nn.Module): The model to train.
        minibatch (dict): A dictionary of tensors from the RolloutBuffer.
        timestep_idx (int): The specific timestep 't' to compute the loss for.
        config (Namespace): The PPO configuration object.

    Returns:
        tuple: A tuple containing (policy_loss, approx_kl, clipfrac, entropy).
    """
    # Unpack tensors from the minibatch
    latents_t = minibatch["latents"][:, timestep_idx]
    old_log_probs = minibatch["old_log_probs"][:, timestep_idx]
    xh_lig, xh_pocket = minibatch["molecules"]
    lig_mask, pocket_mask = minibatch["masks"]
    
    timestep_batch = {
        "molecules": (xh_lig, xh_pocket),
        "masks": (lig_mask, pocket_mask),
        "latents": latents_t,
        "next_latents": minibatch["next_latents"][:, timestep_idx].detach(),
        "timestep": minibatch["timesteps"][:, timestep_idx]
    }
    
    assert_same_ids("compute_loss/masks", lig_mask, pocket_mask)
    # dbg_tensor("compute_loss/old_log_probs", old_log_probs)
    # dbg_tensor("compute_loss/advantages", minibatch['advantages'])
                        
    # Forward pass through the policy network to get new log probabilities.
    #
    # Channel-balanced PPO (optional): the policy log-prob is log_p_pos + log_p_v
    # (continuous coordinates + categorical atom types). The Gaussian coordinate
    # term is orders of magnitude larger in both value and gradient, so without
    # rebalancing it swamps the atom-type term and atom-identity rewards
    # (QED/SA/aromatic) receive almost no effective gradient. config.ppo.channel_grad_scale
    # {pos, v} reweights how much each channel's GRADIENT counts, while leaving the
    # log-prob VALUE (and hence the PPO importance ratio) exactly unchanged. Coordinates
    # are never switched off (s_pos stays > 0) — this is a 3D model and the geometry must
    # keep learning; the knob only stops coords from drowning out atom types.
    scales = getattr(config.ppo, 'channel_grad_scale', None)
    s_pos = float(getattr(scales, 'pos', 1.0)) if scales is not None else 1.0
    s_v   = float(getattr(scales, 'v',   1.0)) if scales is not None else 1.0
    if s_pos == 1.0 and s_v == 1.0:
        # Exact original behaviour (single combined forward).
        new_log_probs = _get_log_probs(policy_network, timestep_batch, config.model.total_timesteps)
    else:
        lp_pos, lp_v = _get_log_prob_channels(
            policy_network, timestep_batch, config.model.total_timesteps)
        new_log_probs = _scale_grad(lp_pos, s_pos) + _scale_grad(lp_v, s_v)
    
    entropy = (-new_log_probs).mean()
    
    # --- PPO Loss Calculation ---
    advantages = minibatch['advantages']
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    clip_range = config.ppo.clip_range
    clipfrac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    
    surr1 = advantages * ratio
    surr2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    
    policy_loss = -torch.min(surr1, surr2).mean()

    # Add entropy bonus
    policy_loss -= config.ppo.entropy_coef * entropy

    # --- KL vs rolling old policy (logging only; kl_coef stays 0.0) ---
    approx_kl = (old_log_probs - new_log_probs).mean()
    kl_coef = getattr(config.ppo, 'kl_coef', 0.0)
    if kl_coef > 0.0:
        policy_loss = policy_loss + kl_coef * approx_kl

    return policy_loss, approx_kl.detach(), clipfrac, entropy

def _get_log_probs(policy_network, timestep_batch, total_timesteps):
    """
    Helper function to compute log π_θ(z_s | z_t, pocket).
    
    Re-indexes per-atom masks to run from 0…n_mol-1 to guarantee they are
    in-range for the sliced tensors.
    """
    
    z_t, z_s = timestep_batch["latents"], timestep_batch["next_latents"]
    xh_lig, xh_pock = timestep_batch["molecules"]
    lig_mask, poc_mask = timestep_batch["masks"]
    t_int = timestep_batch["timestep"].float()
    
    device = z_t.device
    
    # ------------------------------------------------------------------
    # 1) Re-index the *per-atom* masks so they run from 0…n_mol-1.
    #    That guarantees they are in-range for the sliced tensors.
    # ------------------------------------------------------------------
    unique_ids, new_lig_mask = torch.unique(lig_mask, return_inverse=True)
    
    # Build pocket mask through the same mapping
    mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
    mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
    new_poc_mask = mapping[poc_mask]
    
    assert_same_ids("get_log_probs/reindexed", new_lig_mask, new_poc_mask)

    # Quick safety check – will raise before touching CUDA kernels
    assert new_lig_mask.max() < xh_lig.size(0), \
        f"lig_mask out of bounds ({new_lig_mask.max()} ≥ {xh_lig.size(0)})"
    assert new_poc_mask.max() < xh_pock.size(0), \
        f"poc_mask out of bounds ({new_poc_mask.max()} ≥ {xh_pock.size(0)})"

    # ------------------------------------------------------------------
    # 2) Normalise timestep and delegate to the DDPM policy network.
    # ------------------------------------------------------------------
    s_int = torch.clamp(t_int - 1, min=0)
    t = (t_int / total_timesteps).unsqueeze(1)
    s = (s_int / total_timesteps).unsqueeze(1)
    
    assert torch.all((t_int == s_int + 1) | (t_int == 0)), \
        "timestep mismatch: t != s+1"

    # All heavy lifting happens inside the policy network
    return policy_network.log_p_zs_given_zt(
        s, t,
        z_t, z_s,
        xh_pock,
        new_lig_mask, new_poc_mask
    )


def _scale_grad(x, s):
    """Straight-through gradient scaling: returns a tensor equal in VALUE to x
    but whose gradient is multiplied by s.

        forward:  s*x + (1-s)*x.detach()  ==  x
        backward: d/dx = s

    Used to reweight a log-prob channel's contribution to the gradient without
    changing the PPO importance ratio (which depends only on the value).
    """
    if s == 1.0:
        return x
    return s * x + (1.0 - s) * x.detach()


def _get_log_prob_channels(policy_network, timestep_batch, total_timesteps):
    """Like _get_log_probs but returns the two additive channels separately:
    (log_p_pos, log_p_v). Re-indexes per-atom masks identically to _get_log_probs
    so the two paths are numerically consistent (their sum equals _get_log_probs)."""
    z_t, z_s = timestep_batch["latents"], timestep_batch["next_latents"]
    xh_lig, xh_pock = timestep_batch["molecules"]
    lig_mask, poc_mask = timestep_batch["masks"]
    t_int = timestep_batch["timestep"].float()
    device = z_t.device

    unique_ids, new_lig_mask = torch.unique(lig_mask, return_inverse=True)
    mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
    mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
    new_poc_mask = mapping[poc_mask]

    s_int = torch.clamp(t_int - 1, min=0)
    t = (t_int / total_timesteps).unsqueeze(1)
    s = (s_int / total_timesteps).unsqueeze(1)

    return policy_network.log_p_zs_given_zt_channels(
        s, t, z_t, z_s, xh_pock, new_lig_mask, new_poc_mask
    )