"""A module for charge density predictions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
from typing import Optional, Tuple
from pymatgen.io.vasp.outputs import VolumetricData

import numpy as np
from numpy.typing import ArrayLike
import torch

from gpnn.system import ChargeDensitySystem
from gpnn.fingerprint import Fingerprint
from gpnn.models import MLP

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"


@dataclass(kw_only=True)
class CIFPredictor:
    """Predict charge density from a CIF file.
    
    Args:
        cif_path (str): Path to the CIF file.
        save_dir (str): Directory to save the CHGCAR file.
        model_path (str): Path to the pretrained model.
        shape (Optional[Tuple[int, int, int]]): Shape of the charge density grid.
        device_str (str): Device to use for prediction.
    Returns:
        None
    """

    cif_path: str
    save_dir: str
    model_path: str
    shape: Optional[Tuple[int, int, int]] = (50, 50, 50)
    device_str: str = "gpu"

    def __post_init__(self) -> None:
        self.features_mean = self.model.hparams.features_mean.to(self.device)
        self.features_std = self.model.hparams.features_std.to(self.device)
        self.targets_mean = self.model.hparams.targets_mean.to(self.device)
        self.targets_std = self.model.hparams.targets_std.to(self.device)

    @property
    def device(self) -> torch.device:
        """Torch device to use for prediction."""
        if self.device_str.lower() == "cpu":
            return torch.device("cpu")
        if self.device_str.lower() == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("GPU not available, using CPU.")
                return torch.device("cpu")
        else:
            raise ValueError(
                f"Invalid device: {self.device_str}. Please choose 'cpu' or 'gpu'."
            )

    @device.setter
    def device(self, value: str) -> None:
        """Set the device with validation."""
        if value.lower() not in ["cpu", "gpu"]:
            raise ValueError(f"Invalid device: {value}. Please choose 'cpu' or 'gpu'.")
        self.device_str = value

    @cached_property
    def system(self) -> ChargeDensitySystem:
        """Load the system from the CIF file."""
        return ChargeDensitySystem.from_cif(self.cif_path, shape=self.shape)

    @cached_property
    def model(self) -> None:
        """Load the pretrained model."""
        model = MLP.load_from_checkpoint(os.path.join(self.model_path, "last.ckpt"))
        model.to(self.device)
        model.eval()
        return model

    def predict(self) -> ArrayLike:
        """Predict the charge density."""

        # Compute the fingerprint
        fingerprint = Fingerprint(
            system=self.system,
            dataset_elements=list(self.model.hparams.dataset_symbols),
            device=self.device_str,
        )

        features = fingerprint.get()

        # Standardize features
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        features = (features - self.features_mean) / self.features_std

        # Predict charge density
        preds = self.model(features)

        # Inverse GPNN standardization
        preds = (preds * self.targets_std) + self.targets_mean

        # Inverse VASP normalization from PyRho
        # (https://github.com/materialsproject/pyrho/blob/2c35912d667e65d7f9d54d63a3693ad6e014a401/src/pyrho/charge_density.py#L489)
        preds = preds.detach().cpu().numpy() * self.system.primitive_structure.volume

        return np.reshape(preds, self.shape)

    def cif_to_chgcar(self) -> ChargeDensitySystem:
        """Create a ChargeDensitySystem from a CIF file."""
        data = self.predict()
        volumetric_data = VolumetricData(
            structure=self.system.primitive_structure, data={"total": data}
        )
        # Prepare save dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, f"{Path(self.cif_path).stem}_gpnn.CHGCAR")
        volumetric_data.write_file(save_path)
