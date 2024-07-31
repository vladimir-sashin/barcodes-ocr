import os
from typing import Literal, Optional

from pydantic import Field

from src.base_config import BaseValidatedConfig, ConfigYamlMixin
from src.constants import TRAIN_CFG
from src.train.config.datamodule_cfg import ClearMLConfig, DataModuleConfig
from src.train.config.lightning_module_cfg import LightningModuleConfig


class TrainerConfig(BaseValidatedConfig):
    min_epochs: int = 7  # prevents early stopping
    max_epochs: int = 20

    # perform a validation loop every N training epochs
    check_val_every_n_epoch: int = 3

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None

    # set True to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool = False

    fast_dev_run: bool = False

    detect_anomaly: bool = False


class VisualizationConfig(BaseValidatedConfig):
    every_n_epochs: int
    log_k_images: int


class VisualizationGlobalConfig(BaseValidatedConfig):
    batches: VisualizationConfig
    preds: VisualizationConfig


class Config(BaseValidatedConfig, ConfigYamlMixin):
    datamodule_cfg: DataModuleConfig
    clearml_cfg: ClearMLConfig = Field(default=ClearMLConfig())
    random_seed: int
    visualization_cfg: VisualizationGlobalConfig
    trainer_cfg: TrainerConfig
    lightning_module_cfg: LightningModuleConfig


def get_train_cfg() -> Config:
    return Config.from_yaml(os.getenv('TRAIN_CFG_PATH', TRAIN_CFG))
