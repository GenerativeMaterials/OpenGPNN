"""Transforms for processing structure files into datasets."""

from datetime import datetime
import os
from pathlib import Path
import time
from typing import Optional, Tuple, Literal, List

import h5py
import numpy as np
from numpy.typing import ArrayLike

from gpnn.system import ChargeDensitySystem
from gpnn.fingerprint import Fingerprint

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

"""This module includes functions for processing structure files into a format suitable for
ingestion into the GPNN model."""


def chgcar_to_h5(
    chgcar_path: str,
    out_dir: str,
    cutoff: float = 6.0,
    shape: Optional[Tuple[int, int, int]] = None,
    downsample_factor: Optional[int] = None,
    device: Literal["cpu", "gpu"] = "gpu",
    override: bool = False,
):
    """Convert a single CHGCAR file to a HDF5 file containing features, targets, and metadata.

    Args:
        chgcar_path (str): Path to CHGCAR file
        out_dir (str): Output directory for processed files
        cutoff (float, optional): Cutoff value for processing. Defaults to 6.0.
        shape (Optional[Tuple[int, int, int]], optional): Shape of the processed data. 
          Defaults to None.
        downsample_factor (Optional[int], optional): Factor by which to downsample data. 
          Defaults to None.
        device (Literal["cpu", "gpu"], optional): Device to use for processing (cpu/gpu). 
          Defaults to "gpu".
        override (bool, optional): Whether to override existing output files. 
          Defaults to False.

    Returns:
        None
    """
    # Track the time taken to build the dataset
    start_time = time.perf_counter()

    # HDF5 file path combines output directory with <_id>.h5
    out_path = str((Path(out_dir) / Path(chgcar_path).stem).with_suffix(".h5"))

    # Terminate or override if file already exists
    if os.path.exists(out_path):
        if not override:
            raise FileExistsError(f"{out_path} already exists")
        os.remove(out_path)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Construct system
    system = ChargeDensitySystem.from_file(chgcar_path, cutoff=cutoff)

    # Resample or downsample if necessary
    system.resample(factor=downsample_factor, new_shape=shape)

    # Compute fingerprint
    fingerprint = Fingerprint(
        system=system,
        cutoff=cutoff,
        device=device,
    )

    # Write targets to HDF5 file
    system.to_hdf5(file_path=out_path)

    # Write features to HDF5 file
    fingerprint.to_hdf5(file_path=out_path)

    end_time = time.perf_counter()

    # Write metadata
    with h5py.File(out_path, "r+") as hdf5_file:
        hdf5_file.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdf5_file.attrs["build_time"] = f"{end_time - start_time: .2f} seconds"
        hdf5_file.attrs["cutoff"] = cutoff
        hdf5_file.attrs["shape"] = str(system.shape)
        hdf5_file.attrs["downsample_factor"] = downsample_factor
        hdf5_file.attrs["n_points"] = hdf5_file["features"].attrs["n_points"]
        hdf5_file.attrs["system_symbols"] = fingerprint.system_symbols
        hdf5_file.attrs["n_features_per_element"] = fingerprint.n_features_per_element
        hdf5_file.attrs["n_features"] = fingerprint.n_features
        hdf5_file.attrs["device"] = device


def match_dataset_symbols(
    features: ArrayLike,
    system_symbols: List[str],
    dataset_symbols: List[str],
    n_features_per_element: int = 32,
) -> ArrayLike:
    """Populate the feature array with zero columns for missing symbols.

    This function adds zero columns to the feature array for symbols that are
    present in the dataset but not in the system. This ensures that the feature
    shape aligns across the full dataset.

    Args:
        features (ArrayLike): Feature array to populate.
        system_symbols (List[str]): Symbols present in the system.
        dataset_symbols (List[str]): Symbols present in the dataset.
        n_features_per_element (int, optional): Number of features per element. Defaults to 32.
    
    Returns:
        ArrayLike: Feature array with zero columns for missing symbols.
    """
    n_features = n_features_per_element * len(dataset_symbols)

    # No reshape is required if the dataset and system symbols are the same
    if set(dataset_symbols) == set(system_symbols):
        return features

    out_arr = np.zeros((features.shape[0], n_features), dtype=np.float32)

    system_start_idx_by_element = {
        s: i * n_features_per_element for i, s in enumerate(system_symbols)
    }

    dataset_start_idx_by_element = {
        s: i * n_features_per_element for i, s in enumerate(dataset_symbols)
    }

    for s in system_symbols:
        system_start_idx = system_start_idx_by_element[s]
        system_end_idx = system_start_idx + n_features_per_element

        dataset_start_idx = dataset_start_idx_by_element[s]
        dataset_end_idx = dataset_start_idx + n_features_per_element

        out_arr[:, dataset_start_idx:dataset_end_idx] = features[
            :, system_start_idx:system_end_idx
        ]

    return out_arr
