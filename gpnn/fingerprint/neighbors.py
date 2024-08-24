"""This module contains the class for computing neighbors of grid points within a cutoff radius."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.optimization import neighbors as nbs

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"


@dataclass(kw_only=True)
class Neighbors:
    """Class for computing neighbors of grid points within a cutoff radius.
    
    Args:
        grid_coords (ArrayLike): Coordinates of grid points.
        atom_coords (ArrayLike): Coordinates of atoms.
        lattice (ArrayLike): Lattice vectors.
        cutoff (float): Cutoff radius for neighbor search.
        pbc (ArrayLike): Periodic boundary conditions. Defaults to [1, 1, 1].
    Returns:
        None
    """
    grid_coords: ArrayLike
    atom_coords: ArrayLike
    lattice: ArrayLike
    cutoff: float
    pbc: ArrayLike = np.array([1, 1, 1], dtype=np.int64)

    def __post_init__(self):
        self.grid_idxs, self.atom_idxs, self.unit_offsets, self.distances = (
            self._get_neigbors()
        )

        self.offset_vectors = self._get_offset_vectors()

    def get(self):
        return self.grid_idxs, self.atom_idxs, self.distances, self.offset_vectors

    def _get_neigbors(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Find the neighbors of each grid point within the cutoff radius."""
        grid_idxs, atom_idxs, unit_offsets, distances = nbs.find_points_in_spheres(
            all_coords=self.atom_coords,
            center_coords=self.grid_coords,
            r=self.cutoff,
            pbc=self.pbc,
            lattice=self.lattice,
        )
        return grid_idxs, atom_idxs, unit_offsets, distances

    def _get_offset_vectors(self) -> ArrayLike:
        """Compute the displacement vectors between atoms and grid points."""
        # Get the coordinates of atoms and grid points using indices
        atom_coords = self.atom_coords[self.atom_idxs]
        grid_coords = self.grid_coords[self.grid_idxs]

        # Compute the translation vectors for periodic boundary conditions
        translation_vectors = self.unit_offsets @ self.lattice

        # Adjust grid coordinates for periodic boundary conditions
        atom_coords += translation_vectors

        # Compute displacement along x, y, and z axes
        offset_vectors = atom_coords - grid_coords

        return offset_vectors
