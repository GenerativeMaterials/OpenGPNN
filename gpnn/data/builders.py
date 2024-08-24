"""Dataset builders for charge density systems."""

import itertools
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm.auto import tqdm
from pymatgen.core import Element

from gpnn.fingerprint import Fingerprint
from gpnn.system import ChargeDensitySystem


__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

"""This module contains classes for building datasets from charge density systems."""


@dataclass
class DictSplits:
    """Orgnanizes dataset structures into appropriate splits.
    
    Args:
        structure_ids (List[str]): List of structure names.
        splits (Union[List[int], List[float], Dict[str, List[str]]): The splits to
          use for the dataset. This can be a list of integers, a list of floats, or a
          dictionary of lists of strings. If a list of integers or floats is provided,
          the splits will be assigned based on the proportion of the dataset that each
          split should contain. If a dictionary is provided, the keys should be the
          split names and the values should be lists of structure names.
    """

    structure_ids: List[str]
    splits: Union[List[int], List[float], Dict[str, List[str]]]
    _dict_splits: Dict[str, List[str]] = field(init=False)

    def __post_init__(self):
        self._dict_splits = self._process_splits(self.splits)
        self._validate_dict_splits()

    @property
    def train(self) -> List[str]:
        """Return the IDs in the training split."""
        return self._dict_splits.get("train", [])

    @property
    def val(self) -> List[str]:
        """Return the IDs in the validation split."""
        return self._dict_splits.get("val", [])

    @property
    def test(self) -> List[str]:
        """Return the IDs in the test split."""
        return self._dict_splits.get("test", [])

    def _validate_dict_splits(self):
        """Validate the dictionary of splits."""
        assert set(self._dict_splits.keys()) <= {"train", "val", "test"}
        assert all(isinstance(x, str) for v in self._dict_splits.values() for x in v)
        assert all(
            len(set(v)) == len(v) for v in self._dict_splits.values()
        ), "All splits must contain unique structure names."
        assert len(
            set(string for values in self._dict_splits.values() for string in values)
        ) == sum(
            len(values) for values in self._dict_splits.values()
        ), "All splits must be disjoint."

    def _process_splits(
        self, splits: Union[List[int], List[float], Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        if isinstance(splits, list):
            if all(isinstance(x, float) for x in splits):
                return self._assign_splits_from_floats(splits)
            if all(isinstance(x, int) for x in splits):
                return self._assign_splits_from_ints(splits)
            raise ValueError(
                "All elements in the splits list must be of the same type "
                "(either all int or all float)."
            )
        if isinstance(splits, dict):
            return splits
        else:
            raise TypeError(
                "splits must be either a list of integers, a list of floats, "
                "or a dictionary of lists of strings."
            )

    def _assign_splits_from_floats(self, splits: List[float]) -> Dict[str, List[str]]:
        """Assign splits based on proportions."""
        if not math.isclose(sum(splits), 1.0):
            raise ValueError("The sum of the splits proportions must be 1.")

        n = len(self.structure_ids)
        indices = list(range(n))

        split_indices = {
            "train": indices[: int(n * splits[0])],
            "val": indices[int(n * splits[0]) : int(n * (splits[0] + splits[1]))],
            "test": indices[int(n * (splits[0] + splits[1])) :],
        }

        return {
            k: [self.structure_ids[i] for i in v] for k, v in split_indices.items()
        }

    def _assign_splits_from_ints(self, splits: List[int]) -> Dict[str, List[str]]:
        """Assign splits based on counts."""
        if sum(splits) > len(self.structure_ids):
            raise ValueError(
                "The sum of the splits must be less than or equal to the number "
                "of structure names."
            )
        iterator = iter(self.structure_ids)
        return {
            "train": list(itertools.islice(iterator, splits[0])),
            "val": list(itertools.islice(iterator, splits[1])),
            "test": list(itertools.islice(iterator, splits[2])),
        }

    def __getitem__(self, key):
        """Allows access to splits using dictionary syntax."""
        return self._dict_splits.get(key, None)


@dataclass(kw_only=True)
class StructurePathHandler:
    """Paths to individual structure files and their ids.
    
    Args:
        structure_paths (List[str]): List of paths to structure files.
        structure_ids (List[str]): List of structure ids derived from filenames.
    """

    structure_paths: List[str]
    structure_ids: List[str]

    @classmethod
    def from_chgcars(cls, in_dir: str):
        """Initialize from CHGCAR files."""
        paths = []
        ids = []
        for path in Path(in_dir).rglob("*.CHGCAR"):
            paths.append(str(path))
            ids.append(str(path.stem))

        return cls(
            structure_paths=paths,
            structure_ids=ids,
        )


@dataclass(kw_only=True)
class HDF5DatasetBuilder:
    """Class responsible for building HDF5 dataset files."""

    in_dir: str
    out_dir: str
    filename: str
    cutoff: float = 6.0
    shape: Optional[Tuple[int, int, int]] = None
    downsample_factor: Optional[int] = None
    dataset_elements: Sequence[int] = None
    splits: Union[List[int], List[float], Dict[str, List[str]]]
    device: Literal["cpu", "gpu"] = "cpu"

    def __post_init__(self):
        self.out_path = os.path.join(self.out_dir, self.filename)
        self.path_handler = StructurePathHandler.from_chgcars(in_dir=self.in_dir)
        self.structure_paths = self.path_handler.structure_paths
        self.structure_ids = self.path_handler.structure_ids
        self.dict_splits = DictSplits(
            structure_ids=self.structure_ids, splits=self.splits
        )
        self.build_time = None  # Populated when the dataset is built.

        # Elements included in the dataset sorted by electronegativity.
        self.dataset_elements = sorted(
            [
                Element(e) if isinstance(e, str) else Element.from_Z(e)
                for e in self.dataset_elements
            ]
        )

    def _to_hdf5(self) -> None:
        """Compute the features then write the features and targets to an HDF5 file."""
        start_time = time.perf_counter()

        for structure_path, structure_id in tqdm(
            zip(self.structure_paths, self.structure_ids),
            total=len(self.structure_paths),
            desc="Building dataset",
            unit="structure",
            leave=False,
        ):
            # Construct system
            system = ChargeDensitySystem.from_file(structure_path, cutoff=self.cutoff)

            # Resample or downsample if necessary
            system.resample(factor=self.downsample_factor, new_shape=self.shape)

            # Compute fingerprint
            fingerprint = Fingerprint(
                system=system,
                cutoff=self.cutoff,
                dataset_elements=self.dataset_elements,
                device=self.device,
            )

            # Write targets to HDF5 file
            system.to_hdf5(file_path=self.out_path, group=structure_id)

            # Write features to HDF5 file
            fingerprint.to_hdf5(file_path=self.out_path, group=structure_id)

        end_time = time.perf_counter()
        self.build_time = end_time - start_time

    def build(self, override: bool = False) -> None:
        """Compose the dataset file."""
        out_path = Path(self.out_path)

        # Terminate or override if the dataset file already exists
        if out_path.exists():
            if override:
                out_path.unlink()  # remove existing file
            else:
                raise FileExistsError(
                    f"This dataset already exists at {out_path}.\n"
                    "Please remove it or choose a different filename."
                )
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        self._to_hdf5()  # Compute and write features and targets
        self._assign_splits()  # Using metadata flags ["train", "val", "test"]
        self._compute_mean_std()  # For feature and target standardization
        self._write_metadata()  # To track dataset properties

    def _assign_splits(self) -> None:
        """Assign splits to HDF5 groups using metadata."""
        with h5py.File(self.out_path, "r+") as hdf5_file:
            for structure_id in self.structure_ids:
                hdf5_file[structure_id].attrs["split"] = self._get_split(
                    structure_id
                )

    def _get_split(self, structure_id: str) -> str:
        """Determine which split the structure belongs to."""
        if structure_id in self.dict_splits.train:
            return "train"
        if structure_id in self.dict_splits.val:
            return "val"
        if structure_id in self.dict_splits.test:
            return "test"
        raise ValueError(
            f"Structure {structure_id} does not belong to any split."
        )

    def _compute_mean_std(self):
        """Compute mean and std for features and targets following Welford's method."""
        # Initialize aggregates for features
        total_features_sum = 0
        total_features_sum_of_squares = 0
        total_features_points = 0

        # Initialize aggregates for targets
        total_targets_sum = 0
        total_targets_sum_of_squares = 0
        total_targets_points = 0

        with h5py.File(self.out_path, "r") as hdf5_file:
            for structure_id in self.dict_splits.train:
                # Aggregate for features
                features_dataset = hdf5_file[f"{structure_id}/features"]
                total_features_sum += features_dataset.attrs["sum"]
                total_features_sum_of_squares += features_dataset.attrs[
                    "sum_of_squares"
                ]
                total_features_points += features_dataset.attrs["n_points"]

                # Aggregate for targets
                targets_dataset = hdf5_file[f"{structure_id}/targets"]
                total_targets_sum += targets_dataset.attrs["sum"]
                total_targets_sum_of_squares += targets_dataset.attrs["sum_of_squares"]
                total_targets_points += targets_dataset.attrs["n_points"]

        # Compute the overall mean and std for features
        features_mean = total_features_sum / total_features_points
        features_std = np.sqrt(
            total_features_sum_of_squares / total_features_points
            - np.square(features_mean)
        )

        # Compute the overall mean and std for targets
        targets_mean = total_targets_sum / total_targets_points
        targets_std = np.sqrt(
            total_targets_sum_of_squares / total_targets_points
            - np.square(targets_mean)
        )

        # Add computed values as attributes to the root group of the HDF5 file
        with h5py.File(self.out_path, "r+") as hdf5_file:
            hdf5_file.attrs["features_mean"] = features_mean.astype(np.float32)
            hdf5_file.attrs["features_std"] = features_std.astype(np.float32)
            hdf5_file.attrs["targets_mean"] = targets_mean.astype(np.float32)
            hdf5_file.attrs["targets_std"] = targets_std.astype(np.float32)

    def _write_metadata(self) -> None:
        """Write metadata to HDF5 group."""
        # Collect dataset wide information
        metadata = {
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "build_time": self.build_time if self.build_time else "NA",
            "cutoff": self.cutoff,
            "shape": str(self.shape),
            "downsample_factor": str(self.downsample_factor),
            # Elements must be stored as strings for HDF5 compatibility
            "dataset_symbols": [element.symbol for element in self.dataset_elements],
            "n_structures": len(self.structure_ids),
            "device": self.device,
        }
        with h5py.File(self.out_path, "r+") as hdf5_file:
            # Collect shape information from example structure. This will be
            # consistent across all structures in the dataset.
            example_structure = hdf5_file[self.structure_ids[0]]
            metadata["n_points"] = example_structure["features"].shape[0]
            metadata["n_features"] = example_structure["features"].shape[1]
            metadata["n_targets"] = example_structure["targets"].shape[1]

            # Write metadata to HDF5 file
            for key, value in metadata.items():
                hdf5_file.attrs[key] = value
