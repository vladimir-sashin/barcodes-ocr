from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.constants import MONITOR_METRIC
from src.train.config.train_config import Config
from src.train.train_utils.callbacks.draw_callbacks import (
    VisualizeBatch,
    VisualizePreds,
)
from src.train.train_utils.callbacks.model_callbacks import (
    ExportONNX,
    LogModelSummary,
)


def get_callbacks(cfg: Config) -> list[Callback]:
    callbacks = [
        ModelCheckpoint(
            monitor=MONITOR_METRIC,  # The metric to monitor
            mode='max',  # 'max' for accuracy, 'min' for loss
            save_top_k=1,  # Save the top 1 checkpoint
            every_n_epochs=1,  # Every epoch
            verbose=True,  # Print logs
            save_last=True,  # Save the last checkpoint as well
            filename=f'best-{{epoch}}-string_match={{{MONITOR_METRIC}:.2f}}',  # Name of the checkpoint file
            auto_insert_metric_name=False,  # Because metric name contains '/' char
        ),
        LearningRateMonitor(logging_interval='epoch'),
        VisualizeBatch(**dict(cfg.visualization_cfg.batches)),
        VisualizePreds(**dict(cfg.visualization_cfg.preds)),
    ]

    if not cfg.trainer_cfg.fast_dev_run:
        callbacks.extend([ExportONNX(), LogModelSummary()])

    return callbacks
