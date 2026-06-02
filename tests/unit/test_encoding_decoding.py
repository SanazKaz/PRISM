"""
Sanity tests for DiffSBDD and TargetDiff atom-type encoding/decoding.

Four things verified:

1. TargetDiff 13-class add_aromatic table — every index maps to the correct
   (atomic_number, is_aromatic) pair as defined in transforms.py.

2. DiffSBDD crossdock_full round-trip — atom_encoder and atom_decoder are
   consistent inverses: atom_encoder[atom_decoder[i]] == i for all i.

3. Cross-contamination guard — the same numeric index decodes to a DIFFERENT
   element under TargetDiff vs DiffSBDD, proving the two pipelines are not
   interoperable and mixing them would produce silently wrong molecules.

4. Regression: get_atomic_number_from_index requires a tensor, not a list —
   passing a Python list raises AttributeError because the function calls
   .tolist() internally. This was the root cause of 100% reconstruction
   failure during TargetDiff PPO training.
"""

import sys
import pytest
import torch
from pathlib import Path

_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
_DIFFSBDD_ROOT   = _PROJECT_ROOT / "src" / "models" / "diffsbdd"
_TARGETDIFF_ROOT = _PROJECT_ROOT / "src" / "models" / "targetdiff"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_DIFFSBDD_ROOT))
# TargetDiff inserted last so DiffSBDD's flat utils.py wins when both roots
# are present. Tests that need TargetDiff's utils/ package call
# _ensure_targetdiff_utils() explicitly to reprioritise.
sys.path.insert(2, str(_TARGETDIFF_ROOT))


# ---------------------------------------------------------------------------
# Test 1 — TargetDiff 13-class add_aromatic table
# ---------------------------------------------------------------------------

# Ground truth taken directly from MAP_ATOM_TYPE_AROMATIC_TO_INDEX in
# src/models/targetdiff/utils/transforms.py.  If the vendor updates this
# table the test will catch any mismatch.
_EXPECTED_AROMATIC_TABLE = [
    # idx  atomic_num  is_aromatic  element
    (0,    1,          False),   # H
    (1,    6,          False),   # C
    (2,    6,          True),    # C aromatic
    (3,    7,          False),   # N
    (4,    7,          True),    # N aromatic
    (5,    8,          False),   # O
    (6,    8,          True),    # O aromatic
    (7,    9,          False),   # F
    (8,    15,         False),   # P
    (9,    15,         True),    # P aromatic
    (10,   16,         False),   # S
    (11,   16,         True),    # S aromatic
    (12,   17,         False),   # Cl
]


def _load_targetdiff_transforms():
    """Import TargetDiff transforms, handling the utils namespace collision."""
    from src.prism.models.targetdiff_inference import _ensure_targetdiff_utils
    _ensure_targetdiff_utils()
    from utils import transforms as trans  # noqa: E402
    return trans


def test_targetdiff_add_aromatic_table():
    """Every index in the 13-class add_aromatic encoding maps to the correct
    (atomic_number, is_aromatic) pair."""
    trans = _load_targetdiff_transforms()

    for idx, expected_atomic_num, expected_is_aromatic in _EXPECTED_AROMATIC_TABLE:
        t = torch.tensor([idx])
        atomic_nums = trans.get_atomic_number_from_index(t, mode='add_aromatic')
        is_aromatics = trans.is_aromatic_from_index(t, mode='add_aromatic')

        assert atomic_nums == [expected_atomic_num], (
            f"Index {idx}: expected atomic_num={expected_atomic_num}, "
            f"got {atomic_nums}"
        )
        assert is_aromatics == [expected_is_aromatic], (
            f"Index {idx}: expected is_aromatic={expected_is_aromatic}, "
            f"got {is_aromatics}"
        )


def test_targetdiff_add_aromatic_table_has_13_entries():
    """Exactly 13 classes — any addition or removal breaks PPO rollout shapes."""
    trans = _load_targetdiff_transforms()
    # FeaturizeProteinAtom.num_atom_types returns len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
    from utils.transforms import MAP_ATOM_TYPE_AROMATIC_TO_INDEX  # noqa: E402
    assert len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX) == 13, (
        f"Expected 13 add_aromatic classes, got {len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — DiffSBDD crossdock_full encoder/decoder round-trip
# ---------------------------------------------------------------------------

def test_diffsbdd_encoder_decoder_roundtrip():
    """atom_encoder and atom_decoder from crossdock_full are consistent inverses."""
    from constants import dataset_params  # noqa: E402
    info = dataset_params['crossdock_full']
    encoder = info['atom_encoder']
    decoder = info['atom_decoder']

    # decoder → encoder: every element in the decoder must map back to its index.
    for i, element in enumerate(decoder):
        assert encoder[element] == i, (
            f"Round-trip failure: decoder[{i}]='{element}' but "
            f"encoder['{element}']={encoder[element]}"
        )

    # encoder → decoder: every element in the encoder must appear in the decoder.
    for element, idx in encoder.items():
        assert decoder[idx] == element, (
            f"Round-trip failure: encoder['{element}']={idx} but "
            f"decoder[{idx}]='{decoder[idx]}'"
        )


def test_diffsbdd_crossdock_full_has_10_atom_types():
    """Exactly 10 atom types in crossdock_full (C,N,O,S,B,Br,Cl,P,I,F)."""
    from constants import dataset_params  # noqa: E402
    info = dataset_params['crossdock_full']
    assert len(info['atom_decoder']) == 10
    assert set(info['atom_decoder']) == {'C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'}


# ---------------------------------------------------------------------------
# Test 3 — Cross-contamination guard
# ---------------------------------------------------------------------------

# Pairs where TargetDiff and DiffSBDD decode the SAME index to DIFFERENT things.
# Columns: (index, td_atomic_num, diffsbdd_element_at_same_index)
# For indices >= 10 DiffSBDD raises IndexError (only 0-9 exist).
_CROSS_CONTAMINATION_CASES = [
    # idx  TD atomic_num   DiffSBDD element at same idx
    (0,    1,              'C'),   # TD=H,        DSBD=C
    (1,    6,              'N'),   # TD=C,        DSBD=N  (both carbon vs nitrogen!)
    (2,    6,              'O'),   # TD=C(ar),    DSBD=O
    (3,    7,              'S'),   # TD=N,        DSBD=S
    (5,    8,              'B'),   # TD=O,        DSBD=B
    (7,    9,              'P'),   # TD=F,        DSBD=P
    (9,    15,             'F'),   # TD=P(ar),    DSBD=F
]


def test_cross_contamination_guard():
    """TargetDiff and DiffSBDD decode the same numeric index to different elements.

    This proves that using one model's decoder for the other's outputs would
    produce silently wrong molecules — the pipelines are not interchangeable.
    """
    trans = _load_targetdiff_transforms()

    from constants import dataset_params  # noqa: E402
    diffsbdd_decoder = dataset_params['crossdock_full']['atom_decoder']

    for idx, td_atomic_num, dsbd_element in _CROSS_CONTAMINATION_CASES:
        t = torch.tensor([idx])
        td_decoded = trans.get_atomic_number_from_index(t, mode='add_aromatic')[0]
        dsbd_decoded_element = diffsbdd_decoder[idx]

        # TargetDiff decodes to the expected atomic number.
        assert td_decoded == td_atomic_num, (
            f"Index {idx}: TargetDiff atomic_num expected {td_atomic_num}, got {td_decoded}"
        )

        # The two decoders produce DIFFERENT elements for the same index.
        # Map td_atomic_num to element symbol for a meaningful comparison.
        _ATOMIC_NUM_TO_SYMBOL = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F',
                                  15:'P', 16:'S', 17:'Cl'}
        td_symbol = _ATOMIC_NUM_TO_SYMBOL[td_atomic_num]
        assert td_symbol != dsbd_decoded_element, (
            f"Index {idx}: TargetDiff and DiffSBDD BOTH decode to '{td_symbol}' — "
            f"this would hide cross-contamination. Update the test case table."
        )


def test_targetdiff_indices_overflow_diffsbdd_decoder():
    """TargetDiff indices 10-12 are out of range for DiffSBDD's 10-class decoder.

    Feeding TargetDiff rollout tensors into DiffSBDD's build_molecule would
    raise an IndexError for any atom typed as index >= 10.
    """
    from constants import dataset_params  # noqa: E402
    diffsbdd_decoder = dataset_params['crossdock_full']['atom_decoder']

    for td_idx in (10, 11, 12):  # S, S(ar), Cl in TargetDiff encoding
        with pytest.raises(IndexError):
            _ = diffsbdd_decoder[td_idx]


# ---------------------------------------------------------------------------
# Test 4 — Regression: transforms require a tensor, not a Python list
# ---------------------------------------------------------------------------

def test_targetdiff_transforms_require_tensor_not_list():
    """get_atomic_number_from_index calls .tolist() on its argument.

    Passing a Python list instead of a torch.Tensor raises AttributeError
    because list has no .tolist() method. This was the root cause of 100%
    reconstruction failure during TargetDiff PPO training: _reconstruct was
    converting atom_indices to a Python list before passing to these functions.

    The fix: pass atom_indices directly as the tensor it already is.
    """
    trans = _load_targetdiff_transforms()

    # Tensor: must work and return correct atomic numbers.
    indices_tensor = torch.tensor([1, 3, 5])  # C, N, O in add_aromatic
    result = trans.get_atomic_number_from_index(indices_tensor, mode='add_aromatic')
    assert result == [6, 7, 8], f"Unexpected atomic numbers: {result}"

    aromatic = trans.is_aromatic_from_index(indices_tensor, mode='add_aromatic')
    assert aromatic == [False, False, False], f"Unexpected aromaticity: {aromatic}"

    # Python list: must raise AttributeError (list has no .tolist()).
    with pytest.raises(AttributeError, match="tolist"):
        trans.get_atomic_number_from_index([1, 3, 5], mode='add_aromatic')

    with pytest.raises(AttributeError, match="tolist"):
        trans.is_aromatic_from_index([1, 3, 5], mode='add_aromatic')


def test_reconstruction_fn_closure_uses_tensor_path():
    """make_targetdiff_reconstruction_fn returns a closure that passes atom_indices
    directly to trans functions as a tensor, not converting to list first.

    Verify by inspecting that a tensor input reaches trans without AttributeError.
    If the regression were reintroduced (atom_indices.tolist() called before passing),
    the inner trans call would receive a list and raise AttributeError.
    """
    from src.prism.models.targetdiff_inference import make_targetdiff_reconstruction_fn
    _reconstruct = make_targetdiff_reconstruction_fn()

    # A minimal valid benzene-like tensor: 6 aromatic carbons (index 2)
    # arranged at ring positions. Reconstruction may fail (MolReconsError is OK),
    # but it must NOT raise AttributeError — that would mean atom_indices was
    # converted to a list before being passed to trans functions.
    import math
    n = 6
    coords = torch.tensor([
        [math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), 0.0]
        for i in range(n)
    ])
    atom_indices = torch.tensor([2] * n)  # all C aromatic

    try:
        _reconstruct(coords, atom_indices)
        # Success or None return are both fine.
    except AttributeError as e:
        pytest.fail(
            f"_reconstruct raised AttributeError — atom_indices was likely converted "
            f"to a Python list before being passed to trans functions: {e}"
        )
    except Exception:
        # MolReconsError, OpenBabel errors, etc. are acceptable — the point is
        # that AttributeError (the regression) did NOT fire.
        pass
