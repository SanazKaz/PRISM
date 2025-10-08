# ppo_debug_utils.py
import os, torch
from torch_scatter import scatter_mean

DEBUG_PPO = bool(int(os.getenv("DEBUG_PPO", "1")))   # export DEBUG_PPO=0 to disable

# ---------------------------------------------------------------------
#  Mask & latent invariants
# ---------------------------------------------------------------------
def assert_same_ids(tag: str, mask_a: torch.Tensor, mask_b: torch.Tensor) -> None:
    if not DEBUG_PPO: 
        return
    a, b = torch.unique(mask_a.cpu()), torch.unique(mask_b.cpu())
    if not torch.equal(torch.sort(a).values, torch.sort(b).values):
        raise AssertionError(f"[{tag}] mask ID mismatch:\n  lig={a}\n  poc={b}")

def assert_latent_alignment(tag: str,
                             lat: torch.Tensor,
                             nxt: torch.Tensor,
                             lig_mask: torch.Tensor,
                             poc_mask: torch.Tensor,
                             require_temporal: bool = True) -> None:
    """
    • lat / nxt : (N_atoms, T)  – no batch dim
    • require_temporal=False when you just want shape/mask-consistency
    """
    if not DEBUG_PPO:
        return
    # 1) shape sanity
    if lat.shape != nxt.shape:
        raise AssertionError(f"[{tag}] latent shape mismatch {lat.shape} vs {nxt.shape}")
    # 2) mask consistency
    assert_same_ids(f"{tag}/mask", lig_mask, poc_mask)
    # 3) temporal shift (silent bugs show here)
    if require_temporal:
        # take one molecule to keep it cheap
        first_id = lig_mask[0].item()
        idx = (lig_mask == first_id).nonzero(as_tuple=False)[:, 0]
        seq_lat  = lat[idx]      # (n_atoms, T)
        seq_nxt  = nxt[idx]
        # they should be identical but shifted left by 1
        if not torch.allclose(seq_lat[:, 1:], seq_nxt[:, :-1], atol=1e-6):
            raise AssertionError(f"[{tag}] latents not shifted correctly!")

# ---------------------------------------------------------------------
#  Quick tensor print helpers
# ---------------------------------------------------------------------
def dbg_tensor(tag, tensor, max_elems: int = 5):
    if DEBUG_PPO:
        flat = tensor.flatten()
        print(f"[DBG] {tag}: shape={tuple(tensor.shape)} "
              f"device={tensor.device} dtype={tensor.dtype} "
              f"min={flat.min():.3g} max={flat.max():.3g} "
              f"sample={flat[:max_elems].tolist()}")
# ---------------------------------------------------------------------
#  Minibatch inspection
# ---------------------------------------------------------------------
_seen_mb_ids = set()        # keeps state within a single outer epoch

def reset_seen_mb_ids():
    global _seen_mb_ids
    _seen_mb_ids = set()

def validate_minibatch(mb: dict, tag: str = "mb") -> None:
    """
    Checks a freshly-sliced minibatch for internal consistency.
    • mb["masks"]          – tuple(lig_mask, pocket_mask)
    • mb["rewards"]        – (n_mol,)    (or absent during generation phase)
    • mb["latents"]        – (n_atoms, T)
    """
    if not DEBUG_PPO:
        return

    lig_mask, poc_mask = mb["masks"]
    assert_same_ids(f"{tag}/mask_ids", lig_mask, poc_mask)

    unique_ids = torch.unique(lig_mask.cpu())
    n_mol = len(unique_ids)

    # 1) every mol id appears exactly once across minibatches in the epoch
    global _seen_mb_ids
    overlap = [i for i in unique_ids.tolist() if i in _seen_mb_ids]
    if overlap:
        raise AssertionError(f"[{tag}] duplicate molecule IDs across minibatches: {overlap}")
    _seen_mb_ids.update(unique_ids.tolist())

    # 2) reward / value / advantage lengths match n_mol
    for k in ("rewards", "values", "advantages"):
        if k in mb:
            t = mb[k]
            if t.shape[0] != n_mol:
                raise AssertionError(f"[{tag}] {k}.shape[0] = {t.shape[0]} "
                                     f"≠ n_mol ({n_mol})")

    # 3) latents & next_latents rows equal total atoms selected
    n_atom = lig_mask.numel()
    if mb["latents"].shape[0] != n_atom or mb["next_latents"].shape[0] != n_atom:
        raise AssertionError(f"[{tag}] latent rows {mb['latents'].shape[0]} / "
                             f"{mb['next_latents'].shape[0]} ≠ n_atoms ({n_atom})")

    dbg_tensor(f"{tag}/rewards", mb.get("rewards", torch.tensor([])))
