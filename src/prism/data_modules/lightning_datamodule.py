# src/prism/data/datamodule.py

import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader

from src.prism.data_modules.dataset import ProcessedLigandPocketDataset # Assuming this is in your project's top-level
# import utils # For AppendVirtualNodes if you use it

class LigandPocketDataModule(pl.LightningDataModule):
    """
    Encapsulates all data loading and preparation logic.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datadir = Path(self.config.datadir)
        
        # Data transform for virtual nodes (if used)
        self.data_transform = None
        if getattr(self.config, 'virtual_nodes', False):
            # This logic might need adjustment based on where encoders are defined
            # For now, we assume it's handled outside or passed in config
            print("WARNING: Virtual nodes enabled, ensure encoders are properly configured.")
            # self.data_transform = utils.AppendVirtualNodes(...) 

    def setup(self, stage=None):
        """
        Loads the datasets from disk. Called once per GPU.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ProcessedLigandPocketDataset(
                self.datadir / 'train.npz', transform=self.data_transform
            )
            self.val_dataset = ProcessedLigandPocketDataset(
                self.datadir / 'val.npz', transform=self.data_transform
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ProcessedLigandPocketDataset(
                self.datadir / 'test.npz', transform=self.data_transform
            )

    def train_dataloader(self):
        """
        Standard train DataLoader. For distributed training we use
        torch.utils.data.distributed.DistributedSampler to shard the dataset
        across ranks (no per-epoch cursor logic required).
        """
        # Use a DistributedSampler when running under torch.distributed so
        # each rank gets a disjoint slice of the dataset.
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                shuffle=False
            )
            shuffle = False  # sampler handles sharding; keep order deterministic
        else:
            sampler = None
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True
        )

# class RollingWindowSampler(torch.utils.data.Sampler):
#     """
#     Your custom sampler. It now takes the trainer instance to get the
#     current epoch and rank, keeping it neatly inside the DataModule.
#     """
#     def __init__(self, dataset_size, window, trainer):
#         self.dataset_size = dataset_size
#         self.window = window
#         self.trainer = trainer
#         # NOTE: We no longer calculate the cursor here.

#     def __iter__(self):
#         # Calculate the start index for the CURRENT epoch. This is the only
#         # place this calculation is needed.
#         world_size = self.trainer.world_size
#         rank = self.trainer.global_rank
#         epoch = self.trainer.current_epoch
        
#         start = (rank * self.window + epoch * self.window * world_size) % self.dataset_size
#         end = min(start + self.window, self.dataset_size)
#         return iter(range(start, end))

#     def __len__(self):
#         return self.window