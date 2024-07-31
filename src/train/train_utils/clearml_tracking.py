from pathlib import Path
from typing import Optional

from clearml import OutputModel, Task
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.train.config.train_config import Config
from src.train.logger import LOGGER
from src.train.train_utils.callbacks.model_callbacks import ExportONNX

ARTIFACT_CALLBACKS_CLASSES = (ModelCheckpoint, ExportONNX)


def create_cml_task(cfg: Config) -> Task:
    cml_cfg = cfg.clearml_cfg
    Task.force_requirements_env_freeze()
    return Task.init(
        project_name=cml_cfg.project_name,
        task_name=cml_cfg.experiment_name,
        output_uri=True,  # If `output_uri=True` uses default ClearML output URI
        auto_connect_frameworks={
            'pytorch': False,
            'matplotlib': False,
        },
        reuse_last_task_id=False,
    )


def sync_cfg_with_cml(cfg: Config, task: Task) -> Config:
    cfg_dump = cfg.model_dump()
    task.connect_configuration(configuration=cfg_dump)
    return Config.model_validate(cfg_dump)


def setup_clearml(cfg: Config) -> tuple[Config, Optional[Task]]:
    if cfg.clearml_cfg.track_in_clearml is True and cfg.trainer_cfg.fast_dev_run is False:
        task = create_cml_task(cfg)
        cfg = sync_cfg_with_cml(cfg, task)
        return cfg, task
    return cfg, None


def _get_callback(trainer: Trainer, callback_cls: Callback) -> Callback:
    for callback in trainer.callbacks:
        if isinstance(callback, callback_cls):
            return callback


def track_artifacts(
    cfg: Config,
    task: Optional[Task],
    trainer: Trainer,
) -> None:
    if not task or cfg.trainer_cfg.fast_dev_run is True:
        return

    checkpoint_callback = _get_callback(trainer, ModelCheckpoint)
    export_onnx_callback = _get_callback(trainer, ExportONNX)
    upload_model(
        task,
        Path(checkpoint_callback.best_model_path),
        model_name='best_pth_checkpoint',
        tags=['torch', 'best_checkpoint'],
    )
    upload_model(task, export_onnx_callback.onnx_model_path, model_name='onnx_model', tags=['onnx', 'best_checkpoint'])


def upload_model(
    task: Task,
    checkpoint_path: Optional[Path],
    model_name: str,
    tags: Optional[list[str]] = None,
) -> None:
    if not checkpoint_path:
        LOGGER.info(
            'Skipping uploading of the best %s model to ClearML, because no such checkpoint found.',
            checkpoint_path.suffix,  # type: ignore
        )
        return
    LOGGER.info('Uploading best %s model to ClearML...', checkpoint_path.suffix)
    output_model = OutputModel(task=task, name=model_name, tags=tags)
    output_model.update_weights(weights_filename=str(checkpoint_path), auto_delete_file=False)
