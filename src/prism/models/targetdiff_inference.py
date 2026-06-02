"""
Shared TargetDiff inference helpers.

Used by generate_targetdiff.py, test.py, and test_targets.py.
All functions that depend on TargetDiff internals call _ensure_targetdiff_utils()
first to resolve the utils namespace collision: DiffSBDD ships a flat utils.py
while TargetDiff ships a utils/ package, so they clash when both model roots are
on sys.path at the same time (e.g. test.py which supports both models).
"""

import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

_PROJECT_ROOT    = Path(__file__).resolve().parents[3]
_TARGETDIFF_ROOT = _PROJECT_ROOT / 'src' / 'models' / 'targetdiff'


def _ensure_targetdiff_utils() -> None:
    """Ensure TargetDiff's utils/ package takes precedence in sys.modules.

    DiffSBDD ships a flat utils.py; TargetDiff ships a utils/ package. When
    both roots are on sys.path and DiffSBDD was imported first (e.g. via
    analysis/molecule_builder.py which does `import utils`), Python caches
    the flat module in sys.modules['utils']. Subsequent `from utils.data
    import PDBProtein` then fails with "utils is not a package".

    Fix: move TargetDiff root to the front of sys.path and evict the flat
    module so Python re-imports utils as TargetDiff's package on next access.
    """
    td_root = str(_TARGETDIFF_ROOT)
    if td_root in sys.path:
        sys.path.remove(td_root)
    sys.path.insert(0, td_root)

    utils_mod = sys.modules.get('utils')
    if utils_mod is not None and not hasattr(utils_mod, '__path__'):
        # Flat module (DiffSBDD's utils.py) is cached — evict it and submodules
        for key in [k for k in list(sys.modules) if k == 'utils' or k.startswith('utils.')]:
            del sys.modules[key]


def load_targetdiff_model(checkpoint_path: Path, config_path: Path, device: str):
    """
    Build a TargetDiff ScorePosNet3D from a checkpoint + config via the shared
    policy factory and return it ready for inference.
    """
    from src.prism.utils import dict_to_namespace
    from src.prism.models.policy_factory import build_targetdiff_policy

    with open(config_path) as f:
        config = dict_to_namespace(yaml.safe_load(f))

    policy, _ = build_targetdiff_policy(
        config=config,
        device=torch.device(device),
        warm_start_checkpoint=str(checkpoint_path),
    )
    policy.eval()
    return policy._model


def pocket_from_pdb(pdb_path: str, protein_featurizer) -> "ProteinLigandData":
    """
    Parse a pocket PDB file into the ProteinLigandData format expected by
    sample_diffusion_ligand.
    """
    _ensure_targetdiff_utils()

    import torch as _torch
    from utils.data import PDBProtein                          # noqa: E402
    from datasets.pl_data import ProteinLigandData, torchify_dict  # noqa: E402

    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            "element":      _torch.empty([0], dtype=_torch.long),
            "pos":          _torch.empty([0, 3], dtype=_torch.float),
            "atom_feature": _torch.empty([0, 8], dtype=_torch.float),
            "bond_index":   _torch.empty([2, 0], dtype=_torch.long),
            "bond_type":    _torch.empty([0], dtype=_torch.long),
        },
    )
    return protein_featurizer(data)


def make_targetdiff_reconstruction_fn():
    """
    Returns a reconstruction callable compatible with build_molecules_from_batch.

    The returned function takes (coords: Tensor[N,3], atom_indices: Tensor[N])
    and returns an RDKit Mol or None — the same contract as DiffSBDD's
    build_molecule+process_molecule path, but using TargetDiff's OpenBabel
    pipeline with aromatic information so rewards are computed on correctly
    reconstructed molecules.
    """
    _ensure_targetdiff_utils()

    from utils import transforms as trans   # noqa: E402
    from utils import reconstruct           # noqa: E402

    def _reconstruct(coords, atom_indices):
        try:
            xyz = coords.tolist() if hasattr(coords, 'tolist') else list(coords)
            idx = atom_indices.tolist() if hasattr(atom_indices, 'tolist') else list(atom_indices)
            atomic_nums = trans.get_atomic_number_from_index(idx, mode='add_aromatic')
            aromatic    = trans.is_aromatic_from_index(idx,    mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(
                xyz, atomic_nums, aromatic, basic_mode=False
            )
            from rdkit import Chem
            smi = Chem.MolToSmiles(mol)
            return mol if '.' not in smi else None
        except Exception:
            return None

    return _reconstruct


def reconstruct_molecules(all_pred_pos, all_pred_v, debug=False):
    """
    Convert raw TargetDiff position + atom-type predictions into RDKit molecules.

    Uses TargetDiff's own OpenBabel pipeline with aromatic information
    (basic_mode=False) for correct bond assignment.  Returns a list of the
    same length as all_pred_pos; invalid entries are None.

    Args:
        all_pred_pos: list of [N_atoms, 3] position tensors, one per molecule.
        all_pred_v:   list of [N_atoms] integer atom-type index tensors (add_aromatic).
        debug:        if True, print per-molecule atom-type distribution and errors.
    """
    _ensure_targetdiff_utils()

    import collections
    from rdkit import Chem
    from utils import transforms as trans   # noqa: E402
    from utils import reconstruct           # noqa: E402

    if debug:
        all_v_flat = [int(v) for pv in all_pred_v for v in pv.tolist()]
        counts = collections.Counter(all_v_flat)
        tqdm.write(f"[DEBUG] reconstruct_molecules: {len(all_pred_pos)} mols  "
                   f"atom-type dist (index:count): {dict(sorted(counts.items()))}")

    molecules = []
    error_counts = collections.Counter()

    for mol_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        try:
            if debug:
                tqdm.write(f"[DEBUG]   mol {mol_idx}: {len(pred_v)} atoms  "
                           f"v={pred_v.tolist()}")

            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            pred_aromatic  = trans.is_aromatic_from_index(pred_v, mode="add_aromatic")

            if debug:
                tqdm.write(f"[DEBUG]   mol {mol_idx}: atomic_nums={pred_atom_type}  "
                           f"aromatic={pred_aromatic}")

            # basic_mode=False: pass aromatic flags to OpenBabel for correct bond assignment.
            # Never use basic_mode=True — it silently discards aromaticity.
            mol = reconstruct.reconstruct_from_generated(
                pred_pos, pred_atom_type, pred_aromatic, basic_mode=False, debug=debug, mol_idx=mol_idx
            )
            smiles = Chem.MolToSmiles(mol)

            if "." in smiles:
                if debug:
                    tqdm.write(f"[DEBUG]   mol {mol_idx}: FRAGMENT ({smiles})")
                error_counts["fragment"] += 1
                molecules.append(None)
            else:
                if debug:
                    tqdm.write(f"[DEBUG]   mol {mol_idx}: OK  {smiles}")
                molecules.append(mol)

        except reconstruct.MolReconsError as e:
            error_counts[f"MolReconsError:{e}"] += 1
            if debug:
                tqdm.write(f"[DEBUG]   mol {mol_idx}: MolReconsError: {e}")
            molecules.append(None)
        except Exception as e:
            error_counts[f"{type(e).__name__}:{e}"] += 1
            if debug:
                tqdm.write(f"[DEBUG]   mol {mol_idx}: {type(e).__name__}: {e}")
            molecules.append(None)

    if debug:
        tqdm.write(f"[DEBUG] reconstruct summary: {sum(m is not None for m in molecules)}/{len(molecules)} valid  "
                   f"errors={dict(error_counts)}")

    return molecules
