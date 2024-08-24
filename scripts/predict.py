"""Script for predicting charge density from a directory of CIF files."""

import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from gpnn.predict import CIFPredictor

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "06/20/2024"

"""This script ingests a directory of CIF files and runs model predictions that
are output to a directory as CHGCAR files."""


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function for creating a dataset file from a directory of CIF files."""
    files = list(Path(cfg.inference.cif_dir).rglob("*.cif"))
    for file in tqdm(
        files,
        desc="Processing Files",
        total=len(files),
    ):
        # Initialize dataset builder from hydra configs
        predictor = CIFPredictor(
            cif_path=file,
            save_dir=cfg.inference.save_dir,
            model_path=cfg.inference.model_path,
            shape=cfg.inference.shape,
            device_str=cfg.inference.device,
        )

        predictor.cif_to_chgcar()


if __name__ == "__main__":
    main()
