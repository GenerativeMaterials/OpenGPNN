"""Script for creating a dataset file from a directory of CIF files."""

import hydra
from omegaconf import DictConfig

from gpnn.data.builders import HDF5DatasetBuilder

__author__ = "Jake Vikoren"
__maintainer__ = "Jake Vikoren"
__email__ = "jake.vikoren@genmat.xyz"
__date__ = "06/20/2024"

"""This script ingests a directory of CHGCAR files and composes them into a single HDF5
dataset file that will be used for training or fine-tuning a GPNN model."""

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function for creating a dataset file from a directory of CIF files."""
    # Initialize dataset builder from hydra configs
    builder: HDF5DatasetBuilder = hydra.utils.instantiate(
        cfg.data.builder, _convert_="partial"
    )

    # Build the dataset
    builder.build(cfg.data.override)

if __name__ == "__main__":
    main()
