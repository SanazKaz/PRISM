import torch

def checksum_1d(t: torch.Tensor, n: int = 5) -> torch.Tensor:
    """Cheap invariant hash – sum of first/last n elements (grad-safe)."""
    if t.numel() == 0:
        return torch.tensor(0., device=t.device, dtype=t.dtype)
    a = t.flatten()
    return a[:n].sum() + a[-n:].sum()

def assert_same_ids(tag: str, lig_mask: torch.Tensor, pocket_mask: torch.Tensor):
    """Both masks must contain the *same set* of molecule IDs, ignoring counts."""
    ids1 = torch.unique(lig_mask)
    ids2 = torch.unique(pocket_mask)
    assert torch.equal(ids1, ids2), \
        f"[{tag}] Lig & pocket masks diverged! IDs: {ids1.tolist()} vs {ids2.tolist()}"
    print(f"[PASSED] {tag}: Masks have matching IDs: {ids1.tolist()}")

def assert_atom_mol_match(tag: str, lig_mask, tensor_N):
    """N-length atom tensor must have exactly one row per lig_mask entry."""
    assert lig_mask.shape[0] == tensor_N.shape[0], \
        f"[{tag}] Atom count mismatch: mask {lig_mask.shape[0]} vs tensor {tensor_N.shape[0]}!"
    print(f"[PASSED] {tag}: Atom counts match: {lig_mask.shape[0]}")

def assert_step_match(tag, latents, next_latents):
    """[N,T,F] tensors must share N & T dims."""
    assert latents.shape[:2] == next_latents.shape[:2], \
        f"[{tag}] latents {latents.shape} vs next_latents {next_latents.shape}"
    print(f"[PASSED] {tag}: Tensor shapes match: {latents.shape[:2]}")

def dbg(tag: str, *tensors):
    """Print deterministic tiny hashes – cheap and clutter-free."""
    h = [checksum_1d(t).item() for t in tensors]
    print(f"[ALIGN] {tag} checksums: {h}")