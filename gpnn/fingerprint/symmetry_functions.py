"""Module for computing Gaussian symmetry functions."""

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"

@dataclass(frozen=True)
class GaussianSymmetryFunctions:
    """Class for computing Gaussian symmetry functions.
    
    Args:
        MAX_GAUSSIAN_WIDTH (int): Maximum width of the Gaussian functions.
        N_GAUSSIANS (int): Number of Gaussian functions to use.
        dtype (np.dtype): Data type to use for the Gaussian functions.
    Returns:
        None
    """
    MAX_GAUSSIAN_WIDTH: int = field(default=10)
    N_GAUSSIANS: int = field(default=16)
    dtype: np.dtype = field(default=np.float32)

    @cached_property
    def GAUSSIAN_STANDARD_DEVIATIONS(self):
        return np.geomspace(
            start=0.25, 
            stop=self.MAX_GAUSSIAN_WIDTH, 
            num=self.N_GAUSSIANS,
            dtype=self.dtype,
        )

    @cached_property
    def NORMALIZING_CONSTANTS(self):
        return 1 / (
            ((2 * np.pi) ** (3 / 2)) * (self.GAUSSIAN_STANDARD_DEVIATIONS**3)
        )
