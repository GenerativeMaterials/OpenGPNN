"""Module for training the GPNN model."""

from __future__ import annotations

import os
from pathlib import Path
import hydra

import pytorch_lightning as L
import torch
from omegaconf import DictConfig

from gpnn.models import MLP

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "06/20/2024"

"""This module contains the class responsible for training the GPNN model."""


class Trainer:
    """Handler for training GPNN."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Args:
            cfg (DictConfig): The configuration object composed using hydra. 
        Returns:
            None
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self._setup()

    def _setup(self) -> None:
        # Set seed and device
        if self.cfg.train.deterministic:
            L.seed_everything(self.cfg.train.seed)
        torch.set_float32_matmul_precision("high")

        # Instantiate data module
        self.data_module = hydra.utils.instantiate(self.cfg.train.data_module)

        # Instantiate callbacks
        self.checkpoint_callback = hydra.utils.instantiate(
            self.cfg.train.checkpoint_callback
        )
        self.lr_monitor = hydra.utils.instantiate(self.cfg.train.lr_monitor)

        callbacks = [
            self.checkpoint_callback,
            self.lr_monitor,
        ]

        # Instantiate model
        self._load_model()

        # Instantiate trainer
        self.lightning_trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.train.lightning_trainer,
            callbacks=callbacks,
            logger=None,
        )

    def _load_model(self) -> None:
        """Load model from checkpoint if it exists, otherwise initialize a new model."""
        if Path(self.cfg.train.model_save_dir).exists():
            print("Loading model from checkpoint")
            self.model = MLP.load_from_checkpoint(
                os.path.join(self.cfg.train.model_save_dir, "last.ckpt")
            )
        else:
            self.model = hydra.utils.instantiate(
                self.cfg.model, optim=self.cfg.model.optim, _recursive_=False
            )
        self.model = self.model.to(self.device)

    def fit(self) -> None:
        """Fit model"""
        self.lightning_trainer.fit(self.model, self.data_module)

    def test(self) -> None:
        """Test model."""
        self.lightning_trainer.test(self.model, self.data_module)
