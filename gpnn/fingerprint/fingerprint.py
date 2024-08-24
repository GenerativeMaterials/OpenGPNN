"""Core class for building fingerprints from charge density systems."""

from dataclasses import dataclass
import math
from typing import List, Literal

import cupy as cp
import dask
import dask.array as da
import h5py
import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core import Element
from tqdm.auto import tqdm

from gpnn.fingerprint.neighbors import Neighbors
from gpnn.fingerprint.symmetry_functions import GaussianSymmetryFunctions
from gpnn.system import ChargeDensitySystem

dask.config.set({"array.slicing.split_large_chunks": False})

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"


"""The fingerprint class takes a structure with charge density data (ChargeDensitySystem)
as input and uses it to compute the features that will be input into the ML model. The 
fingerprint is constructed using the list of elements present in the dataset. 
The "system_atomic_numbers" represents the elements present in a given structure and 
"dataset_atomic_numbers" includes all elements in the dataset. The fingerprint is computed
for all elements in the structure and then zero columns are added for elements that are
present in the dataset but not in the structure. This ensures that the feature shape aligns
across the full dataset. The fingerprint is computed in batches to minimize memory issues.
"""


@dataclass
class Fingerprint:
    """Class to processing a ChargeDensitySystem into a feature array.
    
    Args:
        system (ChargeDensitySystem): Charge density system.
        cutoff (float): Cutoff radius for the fingerprint.
        dataset_elements (List[int | str | Element]): Elements to include in the dataset.
        batch_size (int): Batch size for processing grid points.
        device (Literal["cpu", "gpu"]): Device to use for computation.

    Returns:
        None
    """

    system: ChargeDensitySystem
    cutoff: float = 6.0
    dataset_elements: List[int | str | Element] = None
    batch_size: int = 5_000
    device: Literal["cpu", "gpu"] = "cpu"

    def __post_init__(self):
        self.xp = cp if self.device == "gpu" else np
        self._symmetry_functions = GaussianSymmetryFunctions()
        self._prepare_dataset_elements()
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate the input parameters."""
        if self.cutoff <= 0:
            raise ValueError("Cutoff must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        if self.device not in ["cpu", "gpu"]:
            raise ValueError("Device must be either 'cpu' or 'gpu'.")
        if any(
            element not in self.dataset_elements for element in self.system.elements
        ):
            raise ValueError(
                "System elements must be a subset of the dataset elements.\n"
                f"System elements: {self.system_elements}\n"
                f"Dataset elements: {self.dataset_elements}"
            )

        if self.n_dataset_elements < self.n_system_elements:
            raise ValueError(
                "Dataset elements must contain at least as many elements as the material."
            )

    def _prepare_dataset_elements(self):
        """Ensure dataset elements are in the correct format."""
        if not self.dataset_elements:
            # Use the elements present in the given structure
            self.dataset_elements = self.system_elements
        else:
            # Convert from atomic symbol (string) or atomic number (int) to PyMatGen Elements
            self.dataset_elements = [
                (
                    Element(element)
                    if isinstance(element, str)
                    else (
                        Element.from_Z(element) if isinstance(element, int) else element
                    )
                )
                for element in self.dataset_elements
            ]

    @property
    def system_elements(self) -> List[Element]:
        """Elements contained in the system."""
        return self.system.elements

    @property
    def system_symbols(self) -> List[str]:
        """Symbols of the elements in the system."""
        return self.system.symbols

    @property
    def dataset_symbols(self) -> List[str]:
        """Symbols of the elements in the full dataset."""
        return [element.symbol for element in self.dataset_elements]

    @property
    def system_atomic_numbers(self) -> List[int]:
        """Unique atomic numbers in the system."""
        return self.system.atomic_numbers

    @property
    def dataset_atomic_numbers(self) -> List[int]:
        """Unique atomic numbers in the full dataset."""
        return [element.Z for element in self.dataset_elements]

    @property
    def n_system_elements(self) -> int:
        """Number of elements in the system."""
        return len(self.system_elements)

    @property
    def n_dataset_elements(self) -> int:
        """Number of elements in the full dataset."""
        return len(self.dataset_elements)

    @property
    def n_features_per_element(self) -> int:
        """Number of features per element."""
        return 2 * self._symmetry_functions.N_GAUSSIANS

    @property
    def n_points(self) -> int:
        """Number of grid points."""
        return self.system.n_points

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.n_features_per_element * self.n_dataset_elements

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the fingerprint."""
        return (self.n_points, self.n_features)

    @property
    def n_batches(self) -> int:
        """Number of batches."""
        return math.ceil(self.system.n_points / self.batch_size)

    @property
    def chunk_size(self) -> int:
        """Size of chunk."""
        return 1_000

    def get(self) -> ArrayLike:
        """Compute the fingerprint."""
        results = [
            self._process_batch(batch)
            for batch in tqdm(
                self._get_batches(),
                desc="Computing Fingerprint",
                total=self.n_batches,
                unit="batch",
            )
        ]
        return np.concatenate(results, axis=0)

    def to_hdf5(self, file_path: str, group: str = None) -> None:
        """Compute and save the fingerprint to an HDF5 file.

        This method computes the fingerprints in batches and writes to an HDF5 file
        on-the-fly. This is useful for large datasets that may not fit into memory.
        
        Args:
            file_path (str): Path to the HDF5 file.
            group (str): Group within the HDF5 file to save the fingerprint.
            
        Returns:
            None
        """
        with h5py.File(file_path, mode="a") as f:
            dataset = f.require_dataset(
                f"{group}/features",
                shape=(0, self.n_features),
                chunks=(5000, self.n_features),
                maxshape=(None, self.n_features),
                dtype=np.float32,
                compression="gzip",
            )

            # Initialize sum, sum of squares, and count
            sum_values = np.zeros(self.n_features, dtype=np.float32)
            sum_of_squares = np.zeros(self.n_features, dtype=np.float32)
            n_points = 0

            for batch in tqdm(
                self._get_batches(),
                desc="Writing Fingerprint",
                total=self.n_batches,
                unit="batch",
            ):
                # Compute the fingerprint of the batch
                batch = self._process_batch(batch)

                # Update sum, sum of squares, and count with the new batch
                sum_values += np.sum(batch, axis=0)
                sum_of_squares += np.sum(np.square(batch), axis=0)
                n_points += batch.shape[0]

                # Resize the dataset to accommodate the new batch
                new_size = dataset.shape[0] + batch.shape[0]
                dataset.resize(new_size, axis=0)

                # Write the batch to the dataset
                dataset[-batch.shape[0]:] = batch

            # This will be used for on-the-fly mean and std computation
            dataset.attrs["sum"] = sum_values
            dataset.attrs["sum_of_squares"] = sum_of_squares
            dataset.attrs["n_points"] = n_points

    def _get_batches(self) -> List[ArrayLike]:
        """Get the grid coordinates in batches."""
        batches = np.array_split(self.system.grid_coords, self.n_batches)
        return batches

    def _process_batch(self, batch_grid_coords: ArrayLike) -> da.Array:
        """Compute the fingerprint of a single structure."""
        self.batch_size = len(batch_grid_coords)

        # Calculate neighbors, distances, and offset vectors for gridpoint/atom pairs.
        neighbors = Neighbors(
            grid_coords=batch_grid_coords,
            atom_coords=self.system.atom_coords,
            lattice=self.system.lattice,
            cutoff=self.cutoff,
        )

        # The scalar distance array stores the distances between grid points and atoms.
        # This data is used to compute the scalar fingerprint in a broadcasted manner.
        scalar_distance_arr = self._build_scalar_distance_array(
            neighbors.atom_idxs,
            neighbors.grid_idxs,
            neighbors.distances,
            self.chunk_size,
        )

        # We first compute a copy of these distances modified by our cutoff function.
        # This function smoothly decays to zero at the cutoff radius focusing on the
        # region of space near the atom.
        cutoff_scalar_distance_arr = self._cutoff_function(scalar_distance_arr)

        # Now we modify the scalar distances by the Gaussian symmetry functions resulting
        # in the scalar fingerprint array.
        scalar_fingerprint_arr = self._get_scalar_fingerprints(
            scalar_distance_arr,
            cutoff_scalar_distance_arr,
            self._symmetry_functions.NORMALIZING_CONSTANTS,
            self._symmetry_functions.GAUSSIAN_STANDARD_DEVIATIONS,
        )

        # During post processing, we sum over contributions of individual atoms to each
        # grid point. Each species is summed separately and the results are concatenated.
        reduced_scalar_fingerprints = self._postprocess_scalar_fingerprints(
            scalar_fingerprint_arr
        )

        # The above process is repeated for the x, y, and z components of the vector offsets.
        vector_distance_arr = self._build_vector_distance_array(
            neighbors.atom_idxs,
            neighbors.grid_idxs,
            neighbors.offset_vectors,
            self.chunk_size,
        )

        # The scalar distances above are reused to accelerate this portion of the computation.
        vector_fingerprint_arr = self._get_vector_fingerprints(
            vector_distance_arr,
            scalar_fingerprint_arr,
            self._symmetry_functions.GAUSSIAN_STANDARD_DEVIATIONS,
        )

        # Postprocessing is performed to combine the x, y, and z components of the vector
        # fingerprints into a single rotationally invariant vector fingerprint.
        reduced_vector_fingerprints = self._postprocess_vector_fingerprints(
            vector_fingerprint_arr
        )

        # Scalar and vector fingerprints are now combined into a single array.
        fingerprint_array = self._column_block_interleave(
            array1=reduced_scalar_fingerprints,
            array2=reduced_vector_fingerprints,
            block_size=self._symmetry_functions.N_GAUSSIANS,
        ).compute()

        return self._postprocess_full_fingerprints(fingerprint_array)

    def _build_scalar_distance_array(
        self,
        atom_idxs: ArrayLike,
        grid_idxs: ArrayLike,
        distances: ArrayLike,
        chunk_size: int = -1,
    ):
        """Build the scalar distance array."""
        # TODO: consider COO sparse arrays
        # Set scalar array
        scalar_distance_arr = self.xp.zeros((self.batch_size, self.system.n_atoms))
        scalar_distance_arr[grid_idxs, atom_idxs] = distances
        scalar_distance_arr = da.from_array(
            scalar_distance_arr, chunks=(chunk_size, self.system.n_atoms)
        )
        return scalar_distance_arr.astype(self.xp.float32)

    def _build_vector_distance_array(
        self,
        atom_idxs: ArrayLike,
        grid_idxs: ArrayLike,
        offset_vectors: ArrayLike,
        chunk_size: int = -1,
    ):
        """Build the vector distance array."""
        # TODO: consider COO sparse arrays
        # Set vector array
        vector_distance_arr = self.xp.zeros((self.batch_size, self.system.n_atoms, 3))
        vector_distance_arr[grid_idxs, atom_idxs] = offset_vectors
        vector_distance_arr = da.from_array(
            vector_distance_arr,
            chunks=(chunk_size, self.system.n_atoms, 3),
        )

        return vector_distance_arr.astype(self.xp.float32)

    def _get_scalar_fingerprints(
        self,
        scalar_distances: da.Array,
        cutoff_scalar_distances: da.Array,
        NORMALIZING_CONSTANTS: ArrayLike,
        GAUSSIAN_STANDARD_DEVIATIONS: ArrayLike,
    ):
        """Compute the scalar fingerprint array."""

        N_GAUSSIANS = len(GAUSSIAN_STANDARD_DEVIATIONS)
        NORMALIZING_CONSTANTS = self.xp.array(NORMALIZING_CONSTANTS)
        GAUSSIAN_STANDARD_DEVIATIONS = self.xp.array(GAUSSIAN_STANDARD_DEVIATIONS)

        # Shapes to enable broadcasting
        scalar_fingperint_shape =  (self.batch_size, self.system.n_atoms,           1)
        gaussian_component_shape = (              1,                   1, N_GAUSSIANS)

        scalar_distances = da.reshape(scalar_distances, shape=scalar_fingperint_shape)
        cutoff_scalar_distances = da.reshape(
            cutoff_scalar_distances, shape=scalar_fingperint_shape
        )
        NORMALIZING_CONSTANTS = self.xp.reshape(
            NORMALIZING_CONSTANTS, gaussian_component_shape
        )
        GAUSSIAN_STANDARD_DEVIATIONS = self.xp.reshape(
            GAUSSIAN_STANDARD_DEVIATIONS, gaussian_component_shape
        )

        # Set mask to cancel out zero values impacted by the exponential
        mask = scalar_distances != 0
        mask = da.reshape(mask, shape=scalar_fingperint_shape)

        exponential = da.exp(
            (-(scalar_distances**2)) / (2 * GAUSSIAN_STANDARD_DEVIATIONS**2)
        )

        return NORMALIZING_CONSTANTS * (exponential * cutoff_scalar_distances) * mask

    def _get_vector_fingerprints(
        self,
        vector_distances: da.Array,
        scalar_fingerprints: da.Array,
        GAUSSIAN_STANDARD_DEVIATIONS: ArrayLike,
    ):
        """Compute the vector fingerprint array."""

        N_GAUSSIANS = len(GAUSSIAN_STANDARD_DEVIATIONS)
        GAUSSIAN_STANDARD_DEVIATIONS = self.xp.array(GAUSSIAN_STANDARD_DEVIATIONS)

        # Shapes to enable broadcasting
        vector_fingerprints_shape = (self.batch_size, self.system.n_atoms,           1, 3)
        scalar_fingerprints_shape = (self.batch_size, self.system.n_atoms, N_GAUSSIANS, 1)
        gauss_standard_devs_shape = (              1,                   1, N_GAUSSIANS, 1)

        # Apply reshaping
        vector_distances = da.reshape(vector_distances, shape=vector_fingerprints_shape)
        scalar_fingerprints = da.reshape(
            scalar_fingerprints, shape=scalar_fingerprints_shape
        )
        GAUSSIAN_STANDARD_DEVIATIONS = self.xp.reshape(
            GAUSSIAN_STANDARD_DEVIATIONS, gauss_standard_devs_shape
        )

        # Compute the vector fingerprints
        vector_fingerprints = vector_distances * scalar_fingerprints
        return vector_fingerprints / (2 * GAUSSIAN_STANDARD_DEVIATIONS**2)

    def _postprocess_scalar_fingerprints(
        self,
        scalar_fingerprints: da.Array,
    ):
        """Postprocess the scalar fingerprints."""
        reduced = self._hsplit_reduce(scalar_fingerprints)
        return da.concatenate(reduced, axis=1)

    def _postprocess_vector_fingerprints(
        self,
        vector_fingerprints: da.Array,
    ):
        """Postprocess the vector fingerprints."""
        reduced = self._hsplit_reduce(vector_fingerprints)
        combined = da.concatenate(reduced, axis=1)
        return da.sqrt(da.sum(combined**2, axis=-1))  # Invariant

    def _postprocess_full_fingerprints(self, fingerprint_array: ArrayLike):
        """Postprocess the combined fingerprints."""
        if self.device == "gpu":
            fingerprint_array = fingerprint_array.get()

        return self._match_dataset_elements(fingerprint_array)

    def _cutoff_function(self, x: ArrayLike):
        return 0.5 * (da.cos(np.pi * x / self.cutoff) + 1)

    def _hsplit_reduce(self, arr: ArrayLike):
        """Split and sum along the first axis."""
        # Start and end indices for each element in the fingerprint.
        slice_indices = [0, *np.cumsum(list(self.system.element_amount_dict.values()))]
        return [
            arr[:, start:end].sum(axis=1)
            for start, end in zip(slice_indices[:-1], slice_indices[1:])
        ]

    def _match_dataset_elements(self, batch: ArrayLike):
        """Populate the feature array with zero columns for missing elements.

        This function adds zero columns to the feature array for elements that are
        present in the dataset but not in the system. This ensures that the feature
        shape aligns across the full dataset.
        """
        if set(self.dataset_elements) == set(self.system_elements):
            return batch

        out_arr = np.zeros((batch.shape[0], self.n_features), dtype=np.float32)

        system_start_idx_by_atomic_number = {
            z: i * self.n_features_per_element for i, z in enumerate(self.system_atomic_numbers)
        }

        dataset_start_idx_by_atomic_number = {
            z: i * self.n_features_per_element
            for i, z in enumerate(self.dataset_atomic_numbers)
        }

        # Populate appropriate columns with features
        for z in self.system_atomic_numbers:
            system_start_idx = system_start_idx_by_atomic_number[z]
            system_end_idx = system_start_idx + self.n_features_per_element

            dataset_start_idx = dataset_start_idx_by_atomic_number[z]
            dataset_end_idx = dataset_start_idx + self.n_features_per_element

            out_arr[:, dataset_start_idx:dataset_end_idx] = batch[
                :, system_start_idx:system_end_idx
            ]

        return out_arr

    def _column_block_interleave(
        self, array1: ArrayLike, array2: ArrayLike, block_size: int
    ):
        """Interleave the blocks of two arrays along the columns axis"""
        # Ensure the arrays have compatible shapes
        assert array1.shape == array2.shape, "Arrays must have the same shape"

        N, total_columns = array1.shape
        M = total_columns // block_size  # Number of blocks

        # Reshape the arrays to expose the blocks
        reshaped1 = array1.reshape(N, M, block_size)
        reshaped2 = array2.reshape(N, M, block_size)

        # Stack the arrays along a new axis to interleave the blocks
        # The new shape will be (N, M, 2, block_size)
        interleaved = da.stack([reshaped1, reshaped2], axis=2)

        # Reshape back to (N, M * 2 * block_size) to intermix blocks
        result = interleaved.reshape(N, M * 2 * block_size)

        return result
