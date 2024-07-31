import lightning

from src.constants import EXPERIMENTS_DIR
from src.train.config.train_config import Config, get_train_cfg
from src.train.datamodule import OCRDataModule
from src.train.lightning_module import OCRLightningModule
from src.train.train_utils.callbacks.constructor import get_callbacks
from src.train.train_utils.clearml_tracking import (
    setup_clearml,
    track_artifacts,
)


def train(cfg: Config) -> None:
    cfg, task = setup_clearml(cfg)

    lightning.seed_everything(cfg.random_seed)

    datamodule = OCRDataModule(cfg.datamodule_cfg)
    model = OCRLightningModule(cfg.lightning_module_cfg)

    callbacks = get_callbacks(cfg)
    trainer = lightning.Trainer(
        **dict(cfg.trainer_cfg),
        callbacks=callbacks,
        default_root_dir=EXPERIMENTS_DIR,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    # TODO: make Callbacks instead
    track_artifacts(cfg, task, trainer)


if __name__ == '__main__':
    cfg = get_train_cfg()
    train(cfg)
