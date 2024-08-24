"""PyTorch Dataloader class for model training."""

import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from gpnn.data.dataset import HDF5Dataset

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

"""This module contains the HDF5DataModule class, which is used to load data from an HDF5 file. 
Since data is already batched in the HDF5Dataset for computational efficiency, we use a batch 
size of 1 and a custom collate function."""


class HDF5DataModule(LightningDataModule):
    """DataModule for loading data from an HDF5 file.
    
    Args:
        h5_path (str): Path to the HDF5 file.
        batch_size (int): The batch size to use when loading data.
        shuffle (bool): The order of batches is always shuffled. Set to True to
          shuffle samples within each batch.
    """
    def __init__(self, h5_path: str, batch_size: int = 5000, shuffle: bool = True):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset = HDF5Dataset(
                h5_path=self.h5_path,
                split="train",
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
            self.val_dataset = HDF5Dataset(
                h5_path=self.h5_path,
                split="val",
                batch_size=self.batch_size,
                shuffle=False,
            )
        if stage == "test" or stage is None:
            self.test_dataset = HDF5Dataset(
                h5_path=self.h5_path,
                split="test",
                batch_size=self.batch_size,
                shuffle=False,
            )

    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=self.shuffle,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

    def collate_fn(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Avoids adding the batch dimension since our datasets are already batched."""
        return batch[0]
