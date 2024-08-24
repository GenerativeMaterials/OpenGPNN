"""Module defining the core ML model."""

from typing import Any, Dict

import h5py
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "07/11/2024"


class BaseModel(L.LightningModule):
    """Base model class for all models in the project."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Save args and kwargs to self.hparams
        self.save_hyperparameters()

        # When training from scratch, metadata is loaded from the HDF5 dataset
        if not hasattr(self.hparams, 'features_mean'):
            try:
                with h5py.File(self.hparams.h5_path, "r") as file:
                    self.metadata = dict(file.attrs.items())
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Metadata file not found at {self.hparams.h5_path}"
                ) from error

            # Add key info from HDF5 file to hparams
            self.hparams.dataset_symbols = self.metadata["dataset_symbols"]
            self.hparams.input_dim = self.metadata["n_features"]
            self.hparams.output_dim = self.metadata["n_targets"]
            self.hparams.features_mean = torch.tensor(self.metadata["features_mean"])
            self.hparams.features_std = torch.tensor(self.metadata["features_std"])
            self.hparams.targets_mean = torch.tensor(self.metadata["targets_mean"])
            self.hparams.targets_std = torch.tensor(self.metadata["targets_std"])

        # Track mean and std tensors as part of the model state
        self.register_buffer("features_mean", self.hparams.features_mean.clone().detach())
        self.register_buffer("features_std", self.hparams.features_std.clone().detach())
        self.register_buffer("targets_mean", self.hparams.targets_mean.clone().detach())
        self.register_buffer("targets_std", self.hparams.targets_std.clone().detach())

    def inverse_standardize(self, *args: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization to model predictions."""
        # Convert predictions back to original scale
        return ((arg * self.targets_std) + self.targets_mean for arg in args)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.hparams.optim.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.hparams.optim.epochs,
            max_lr=self.hparams.optim.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            "monitor": "val_loss",
        }


class MLP(BaseModel):
    """An MLP capable of predicting charge density values based on local atomic environment.
    
    See GPNN/configs/model/default.yaml for input args.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequential = build_mlp(
            input_dim=self.hparams.input_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_layers=self.hparams.n_layers,
            output_dim=self.hparams.output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.sequential(x)

    def _step(
        self,
        batch: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """Common step for training, validation, and test."""
        features, targets = batch
        preds = self(features)

        # L1 outperforms MSE
        loss = F.l1_loss(preds, targets)

        # Inverse standardize predictions and targets, and calculate RMSE in the input space
        inv_preds, inv_targets = self.inverse_standardize(preds, targets)
        rmse = torch.sqrt(F.mse_loss(inv_preds, inv_targets))

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_rmse", rmse, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch) -> torch.Tensor:
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch) -> torch.Tensor:
        """Validation step."""
        return self._step(batch, "val")

    def test_step(self, batch) -> torch.Tensor:
        """Test step."""
        return self._step(batch, "test")

def build_mlp(
    input_dim: int,
    hidden_dim: int = 300,
    n_layers: int = 3,
    output_dim: int = 1,
):
    """A simple MLP model."""
    # Input layer
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

    # Hidden layers
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

    # Output layer
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)
