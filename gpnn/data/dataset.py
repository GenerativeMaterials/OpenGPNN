"""Module defining the dataset object."""

import random
from pathlib import Path
from typing import List, Literal, Tuple, Union

import h5py
import torch
from torch.utils.data import Dataset

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

"""This module defines the HDF5Dataset class, which is used to load data from an HDF5 file."""


class HDF5Dataset(Dataset):
    """Dataset class for loading data from an HDF5 file."""
    def __init__(
        self,
        h5_path: Union[str, Path],
        split: Literal["train", "val", "test"],
        batch_size: int = 5000,
        shuffle: bool = True,
    ):
        """
        Args:
            h5_path (Union[str, Path]): Path to the HDF5 file.
            split (Literal["train", "val", "test"]): The dataset split to load.
            batch_size (int): The batch size to use when loading data.
            shuffle (bool): The order of batches is always shuffled. Set to True to
              shuffle samples within each batch.
        """
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load structure names and batch info.
        self.structure_names = self._load_structure_names()
        self.batch_info = self._get_batch_info()

        # Load standardization parameters.
        self.features_mean, self.features_std, self.targets_mean, self.targets_std = (
            self._load_mean_std()
        )

    def _get_batch_info(self):
        """Get the start and end indices for the batch."""
        batch_info = []
        with h5py.File(self.h5_path, "r") as file:
            for structure in self.structure_names:
                # Determine the number of data points in the current structure.
                n_points = file[f"{structure}/features"].shape[0]

                # Calculate the number of full batches that can be formed from these grid points.
                n_batches = (n_points + self.batch_size - 1) // self.batch_size

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(
                        start_idx + self.batch_size, n_points
                    )  # Ensure we don't exceed the number of grid points.
                    batch_info.append((structure, start_idx, end_idx))
        random.shuffle(batch_info)  # Always shuffle batches
        return batch_info

    def _load_structure_names(self) -> List[str]:
        """Load structure names for the specified split from the HDF5 file."""
        structure_names = []
        with h5py.File(self.h5_path, "r") as h5_file:
            for structure_name, group in h5_file.items():
                if group.attrs.get("split") == self.split:
                    structure_names.append(structure_name)
        return structure_names

    def _load_mean_std(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load standardization parameters from the HDF5 file."""
        with h5py.File(self.h5_path, "r") as h5_file:
            features_mean = torch.tensor(h5_file.attrs["features_mean"])
            features_std = torch.tensor(h5_file.attrs["features_std"])
            targets_mean = torch.tensor(h5_file.attrs["targets_mean"])
            targets_std = torch.tensor(h5_file.attrs["targets_std"])
        return features_mean, features_std, targets_mean, targets_std

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.structure_names)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a full batch from the given index, standardizing it on the fly."""
        # Get the key, start_idx, and end_idx for the batch.
        structure, start_idx, end_idx = self.batch_info[idx]

        # Load the features and targets from the HDF5 dataset.
        with h5py.File(self.h5_path, "r") as file:
            batch_length = end_idx - start_idx
            shuffled_idxs = (
                torch.randperm(batch_length)
                if self.shuffle
                else torch.arange(batch_length)
            )

            features = torch.tensor(file[f"{structure}/features"][start_idx:end_idx])[
                shuffled_idxs
            ]
            targets = torch.tensor(file[f"{structure}/targets"][start_idx:end_idx])[
                shuffled_idxs
            ]

            # Standardize the features and targets in preparation for training.
            features = (features - self.features_mean) / (self.features_std)
            targets = (targets - self.targets_mean) / (self.targets_std)

        return features, targets
