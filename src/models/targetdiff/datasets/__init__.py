import torch
from torch.utils.data import Subset
try:
    from .pl_pair_dataset import PocketLigandPairDataset
    from .pdbbind import PDBBindDataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if not _DATASETS_AVAILABLE:
        raise ImportError("lmdb is required for dataset loading: pip install lmdb")
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
