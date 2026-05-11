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
                        
    # Forward pass through the policy network to get new log probabilities
    new_log_probs = _get_log_probs(policy_network, timestep_batch, config.model.total_timesteps)
    
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

    # --- KL penalty: E[log π_old - log π_θ] = KL(π_old || π_θ) ---
    # Needs grad so it can propagate; approx_kl is also returned for logging.
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