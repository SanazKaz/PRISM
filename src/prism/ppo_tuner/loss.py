# src/prism/ppo_tuner/loss.py

import torch

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
                        
    # Forward pass through the policy network to get new log probabilities
    new_log_probs = _get_log_probs(policy_network, timestep_batch, config.diffusion_params.diffusion_steps)
    
    entropy = (-new_log_probs).mean()
    
    # --- PPO Loss Calculation ---
    advantages = minibatch['advantages']
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    clip_range = config.ppo_params.clip_range
    clipfrac = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    
    surr1 = advantages * ratio
    surr2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Add entropy bonus
    policy_loss -= config.ppo_params.entropy_coef * entropy
    
    # --- KL Divergence for logging/diagnostics ---
    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
    
    return policy_loss, approx_kl, clipfrac, entropy

def _get_log_probs(policy_network, timestep_batch, total_timesteps):
    """
    Helper function to compute log π_θ(z_s | z_t, pocket).
    """
    z_t, z_s = timestep_batch["latents"], timestep_batch["next_latents"]
    xh_lig, xh_pock = timestep_batch["molecules"]
    lig_mask, poc_mask = timestep_batch["masks"]
    t_int = timestep_batch["timestep"].float()
    
    # Re-index per-atom masks to be contiguous from 0..n_mol-1
    device = z_t.device
    unique_ids, new_lig_mask = torch.unique(lig_mask, return_inverse=True)
    
    # build pocket mask on the correct device
    mapping = -torch.ones(int(poc_mask.max()) + 1, dtype=torch.long, device=device)
    mapping[unique_ids] = torch.arange(len(unique_ids), device=device)
    new_poc_mask = mapping[poc_mask]

    # Normalize timesteps for the diffusion model
    s_int = torch.clamp(t_int - 1, min=0)
    t = (t_int / total_timesteps).unsqueeze(1)
    s = (s_int / total_timesteps).unsqueeze(1)
    
    # The core call to the diffusion model
    return policy_network.log_p_zs_given_zt(
        s, t,
        z_t, z_s,
        xh_pock,
        new_lig_mask, new_poc_mask
    )