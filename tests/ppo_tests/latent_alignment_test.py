# file: ppo_tests/latent_alignment_tests.py
import torch

def _checksum(t: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Grad-safe cheap hash – sum of first/last k elements."""
    if t.numel() == 0:
        return torch.tensor(0., device=t.device, dtype=t.dtype)
    flat = t.flatten()
    return flat[:k].sum() + flat[-k:].sum()

@torch.no_grad()
def assert_latent_alignment(tag: str,
                            latents: torch.Tensor,
                            next_latents: torch.Tensor,
                            lig_mask: torch.Tensor,
                            pocket_mask: torch.Tensor,
                            require_temporal=True):
    # ------- Invariant A
    assert latents.shape == next_latents.shape, \
        f"[{tag}] Shape mismatch: {latents.shape} vs {next_latents.shape}"
    
    # ------- Invariant B
    ids1, ids2 = torch.unique(lig_mask), torch.unique(pocket_mask)
    assert torch.equal(ids1, ids2), \
        f"[{tag}] Mask ID drift! lig={ids1.tolist()}  pocket={ids2.tolist()}"
    
    # ------- Invariant C (+ D when called post-permutation)
    if require_temporal == True:
        diff = (latents[:, 1:] - next_latents[:, :-1]).abs().max()
        assert diff < 1e-6, \
            f"[{tag}] Temporal mis-alignment detected (max|Δ|={diff})"
    
    # ------- Human-readable heartbeat
    ch_lat   = _checksum(latents).item()
    ch_nlat  = _checksum(next_latents).item()
    print(f"[ALIGN✓] {tag}: checksums lat={ch_lat:.3e}  next={ch_nlat:.3e}")
