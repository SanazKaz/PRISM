"""
Shared TargetDiff inference helpers.

Used by generate_targetdiff.py, test.py, and test_targets.py.
All three functions that depend on TargetDiff internals use lazy imports so
this module is safe to import without the TargetDiff root on sys.path.
Callers are responsible for adding _TARGETDIFF_ROOT to sys.path before
invoking pocket_from_pdb or reconstruct_molecules.
"""

from pathlib import Path

import torch
import yaml


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
    sample_diffusion_ligand. Requires _TARGETDIFF_ROOT on sys.path.
    """
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


def reconstruct_molecules(all_pred_pos, all_pred_v) -> list:
    """
    Convert raw TargetDiff position + atom-type predictions into RDKit
    molecules. Returns a list of the same length as all_pred_pos; invalid
    entries are None. Requires _TARGETDIFF_ROOT on sys.path.
    """
    from rdkit import Chem
    from utils import transforms as trans   # noqa: E402
    from utils import reconstruct           # noqa: E402

    molecules = []
    for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
        try:
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode="add_aromatic")
            pred_aromatic  = trans.is_aromatic_from_index(pred_v, mode="add_aromatic")
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            molecules.append(mol if "." not in smiles else None)
        except reconstruct.MolReconsError:
            molecules.append(None)
    return molecules
