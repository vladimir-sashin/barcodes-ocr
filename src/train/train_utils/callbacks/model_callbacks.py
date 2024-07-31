from pathlib import Path
from typing import Optional

import torch
from lightning import Callback, Trainer
from torchinfo import summary

from src.train.lightning_module import OCRLightningModule
from src.train.logger import LOGGER


class LogModelSummary(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:
        images = next(iter(trainer.train_dataloader))[0]

        images = images.to(pl_module.device)
        summary(pl_module.model, input_data=images)


class ExportONNX(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.onnx_model_path: Optional[Path] = None

    def on_test_end(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:
        LOGGER.info('Converting best checkpoint to ONNX...')
        if checkpoint_callback := trainer.checkpoint_callback:
            checkpoint_path = checkpoint_callback.best_model_path
            self.onnx_model_path = _model_to_onnx(trainer, checkpoint_path)
            LOGGER.info('Successfully converted best checkpoint to ONNX.')
        LOGGER.warning(
            "Couldn't convert checkpoint to ONNX because ModelCheckpoint callback wasn't found in Trainer's callbacks.",
        )


def _model_to_onnx(trainer: Trainer, checkpoint_path: str) -> Path:
    data_cfg = trainer.datamodule.data_cfg
    height, width = data_cfg.height, data_cfg.width

    dummy_input = torch.rand(1, 3, height, width)
    torch_model = OCRLightningModule.load_from_checkpoint(checkpoint_path)
    output_path = Path(checkpoint_path).parent / 'model.onnx'
    torch.onnx.export(
        torch_model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': [0], 'output': [0]},
    )

    return output_path
