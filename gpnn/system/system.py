"""A module defining the ChargeDensitySystem object."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import product
from typing import Dict, Tuple, Optional

import copy
import h5py
import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Species
from pymatgen.io.vasp.outputs import VolumetricData
from pyrho.charge_density import ChargeDensity

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

"""The ChargeDensitySystem object is the main representation of atomic strucutres and 
their associated charge density data. This object is used to generate the fingerprints
used for training."""


@dataclass
class ChargeDensitySystem:
    """A class to represent a charge density system.
    
    Args:
        cd (ChargeDensity): The charge density object.
        cutoff (float): The cutoff radius for the supercell.
    Returns:
        None
    """
    cd: ChargeDensity
    cutoff: float = 6.0

    def __post_init__(self):
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.cd, ChargeDensity):
            raise ValueError("cd must be a ChargeDensity object.")
        if not isinstance(self.cutoff, (int, float)):
            raise ValueError("cutoff must be an integer or float.")
        if self.cutoff < 0:
            raise ValueError("cutoff must be a positive number.")

    @cached_property
    def supercell(self) -> Structure:
        """Get the minimal symmetric supercell that contains the cutoff radius."""
        # Orthogonal distance from the center to the walls of the unit cell
        dist_to_pbc = np.diagonal(self.cd.lattice) / 2

        # Compute the number of unit cells needed in each direction
        with np.errstate(divide='ignore'):  # Ignore divide by zero warning
            pbcs_crossed = np.abs((self.cutoff - dist_to_pbc) // (2 * dist_to_pbc))
        pbcs_crossed += 1  # Include the center cell
        return pbcs_crossed.astype(int) * 2 + 1  # convert to symmetrical supercell

    @cached_property
    def primitive_structure(self) -> Structure:
        """Get the primitive structure of the charge density system."""
        return self.cd.structure.get_sorted_structure()

    @cached_property
    def structure(self) -> Structure:
        """Get the structure of the charge density system."""
        structure = self.primitive_structure.copy()
        structure *= self.supercell
        return structure

    @cached_property
    def grid_coords(self) -> ArrayLike:
        """Get the cartesian coordinates of the charge density grid."""
        # Unmodified coords
        primitive_coords = build_coords(
            self.shape, self.cd.lattice.astype(np.float64).tobytes()
        )

        # If the supercell is [1, 1, 1], coords need no translation
        if all(self.supercell == 1):
            return primitive_coords

        # Fractional center
        center_coord = np.array([0.5, 0.5, 0.5])

        # Center in the supercell with a translation vector
        primitive_center = fractional_to_cartesian(
            self.primitive_structure.lattice.matrix, center_coord
        )
        supercell_center = fractional_to_cartesian(self.lattice, center_coord)

        # Translation vector to move coords from the primitive cell to the supercell
        translation_vector = supercell_center - primitive_center
        return primitive_coords + translation_vector

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the charge density grid."""
        return self.cd.grid_shape

    @shape.setter
    def shape(self, new_shape: Tuple[int, int, int]) -> None:
        """Set the shape of the charge density grid."""
        self.resample(new_shape=new_shape)

    @property
    def lattice(self) -> ArrayLike:
        """Get the lattice matrix of the charge density system."""
        return self.structure.lattice.matrix

    @property
    def data(self) -> ArrayLike:
        """Get the charge density data."""
        return self.cd.normalized_data["total"]

    @property
    def atom_coords(self) -> ArrayLike:
        """Get the cartesian atomic coordinates of the charge density system."""
        return self.structure.cart_coords

    @property
    def all_atomic_numbers(self) -> ArrayLike:
        """Get the atomic numbers of all atoms in the charge density system."""
        return self.structure.atomic_numbers

    @property
    def elements(self) -> ArrayLike:
        """Get the elements in the charge density system."""
        elements = self.structure.composition.elements
        if isinstance(elements[0], Species):
            return sorted([s.element for s in elements])
        return elements

    @property
    def symbols(self) -> ArrayLike:
        """Get the chemical symbols of the elements in the charge density system."""
        return [element.symbol for element in self.elements]

    @property
    def atomic_numbers(self) -> ArrayLike:
        """Get the unique atomic numbers of the elements in the charge density system."""
        return [element.Z for element in self.elements]

    @property
    def n_points(self) -> int:
        """Get the number of grid points in the charge density grid."""
        return self.cd.pgrids["total"].ngridpts

    @property
    def n_atoms(self) -> int:
        """Get the number of atoms in the charge density system."""
        return len(self.structure)

    @property
    def n_elements(self) -> int:
        """Get the number of elements in the charge density system."""
        return len(self.elements)

    @property
    def element_amount_dict(self) -> Dict[int, int]:
        """Get the element amount dictionary of the charge density system."""
        return self.structure.composition.get_el_amt_dict()

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the charge density data."""
        return self.data.dtype

    @classmethod
    def from_cube(cls, cube_path: str, cutoff: float = 6.0) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a cube file."""
        cube = VolumetricData.from_cube(cube_path)
        cd = ChargeDensity.from_pmg(cube, normalization=None)
        return cls(cd=cd, cutoff=cutoff)

    @classmethod
    def from_chgcar(cls, chgcar_path: str, cutoff: float = 6.0) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a CHGCAR file."""
        cd = ChargeDensity.from_file(chgcar_path)
        return cls(cd=cd, cutoff=cutoff)

    @classmethod
    def from_cif(
        cls,
        cif_path: str,
        shape: Tuple[int, int, int] = (50, 50, 50),
        cutoff: float = 6.0,
    ) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a CIF file."""
        structure = Structure.from_file(cif_path)
        volumetric_data = VolumetricData(
            structure=structure, data={"total": np.zeros(shape)}
        )
        cd = ChargeDensity.from_pmg(volumetric_data, normalization=None)
        return cls(cd=cd, cutoff=cutoff)

    @classmethod
    def from_file(cls, file_path: str, cutoff: float = 6.0) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a file."""
        if file_path.endswith(".cube.gz"):
            return cls.from_cube(cube_path=file_path, cutoff=cutoff)
        elif file_path.endswith(".CHGCAR"):
            return cls.from_chgcar(chgcar_path=file_path, cutoff=cutoff)
        elif file_path.endswith(".cif"):
            return cls.from_cif(cif_path=file_path, cutoff=cutoff)
        else:
            raise ValueError(
                "File type not supported. Supported types: ['.cube.gz', '.CHGCAR']."
            )

    @classmethod
    def from_structure(
        cls, structure: Structure, data: ArrayLike, cutoff: float = 6.0
    ) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a structure and data."""
        cd = ChargeDensity.from_pmg(
            VolumetricData(structure=structure, data={"total": data}),
            normalization=None,
        )
        return cls(cd=cd, cutoff=cutoff)

    @classmethod
    def from_pmg(
        cls, volumetric_data: VolumetricData, cutoff: float = 6.0
    ) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a pymatgen VolumetricData object."""
        cd = ChargeDensity.from_pmg(volumetric_data, normalization="vasp")
        return cls(cd=cd, cutoff=cutoff)

    def to_hdf5(self, file_path: str, group: str = None) -> None:
        """Save the charge density data to an HDF5 file."""
        data = np.reshape(self.data, (-1, 1))
        chunk_size = 5000 if data.shape[0] > 5000 else data.shape[0]

        # Track for mean and std computation
        sums = np.sum(data, axis=0)
        sum_of_squares = np.sum(data**2, axis=0)
        n_points = data.shape[0]

        if group is not None:
            dataset_path = f"{group}/targets"
        else:
            dataset_path = "targets"

        with h5py.File(file_path, "a") as f:
            dataset = f.create_dataset(
                dataset_path,
                data=data,
                chunks=(chunk_size, 1),
                dtype=np.float32,
                compression="gzip",
            )

            # This will be used for on-the-fly mean and std computation
            dataset.attrs["sum"] = sums
            dataset.attrs["sum_of_squares"] = sum_of_squares
            dataset.attrs["n_points"] = n_points

    def to_dict(self) -> Dict[str, ArrayLike]:
        """Convert the charge density data to a dictionary."""
        data = np.reshape(self.data, (-1, 1))

        # Track for mean and std computation
        sums = np.sum(data, axis=0)
        sum_of_squares = np.sum(data**2, axis=0)
        n_points = data.shape[0]

        metadata = {
            "sum": float(sums[0]),
            "sum_of_squares": float(sum_of_squares[0]),
            "n_points": n_points,
        }

        return {"data": data.tolist(), "metadata": metadata}

    def resample(
            self,
            factor: Optional[int] = 1,
            new_shape: Optional[Tuple[int, int, int]] = None,
        ) -> None:
        """Resample the charge density grid by a factor or to a new shape."""
        if new_shape is None:
            new_shape = [int(s / factor) for s in self.shape]
        self.cd = self.cd.get_transformed(sc_mat=np.eye(3), grid_out=new_shape)

    def copy(self) -> ChargeDensitySystem:
        return copy.deepcopy(self)


@lru_cache
def build_coords(shape: Tuple[int, int, int], lattice: bytes) -> ArrayLike:
    """Converts a grid shape and lattice into a set of coordinates.

    The lattice is expected in bytes to enable caching. This avoids expensive
    calls to this function when the grid coordinates are unchanged.

    Args:
        shape (tuple): The shape of the grid.
        lattice (bytes): The lattice vectors of the grid.

    Returns:
        ArrayLike: The cartesian coordinates of the grid.
    """

    # Convert bytes back to numpy array
    lattice = np.frombuffer(lattice, dtype=np.float64).reshape((3, 3))

    # Fractional values along each cartesian axis
    axes = [np.linspace(0, 1, n, endpoint=False) for n in shape]

    # Generate the fractional coordinates
    fractional_coords = np.array(list(product(*axes)))

    # Convert fractional to cartesian coordinates
    return fractional_coords.dot(lattice)


def fractional_to_cartesian(lattice_matrix, fractional_coords):
    """Convert fractional coordinates to cartesian coordinates."""
    return np.matmul(lattice_matrix.T, fractional_coords.T).T
