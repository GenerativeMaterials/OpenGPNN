import os

import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from gpnn.train.trainer import Trainer

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.train.seed)

    # Run training
    trainer: Trainer = hydra.utils.instantiate(
        cfg.train.trainer, cfg=cfg, _recursive_=False
    )
    trainer.fit()
    trainer.test()

if __name__ == "__main__":
    main()
