import torch
from lightning import Callback, Trainer
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

from src.constants import PRED_TEXT_SIZE
from src.train.data_utils.image_convertation import (
    cv_image_to_tensor,
    denormalize,
    tensor_to_cv_image,
)
from src.train.lightning_module import OCRLightningModule
from src.train.train_utils.predict_utils import matrix_to_string

TRAIN_STAGE = 'train'


class VisualizeCallback(Callback):
    def __init__(self, every_n_epochs: int, log_k_images: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.log_k_images = log_k_images


class VisualizePreds(VisualizeCallback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:
        self._log_batch(trainer, pl_module, stage=TRAIN_STAGE)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:
        self._log_batch(trainer, pl_module, stage='val')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:
        self._log_batch(trainer, pl_module, stage='test')

    def _log_batch(self, trainer: Trainer, pl_module: OCRLightningModule, stage: str) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        dataloader_name = f'{stage}_dataloader' if stage == TRAIN_STAGE else f'{stage}_dataloaders'
        dataloader = getattr(trainer, dataloader_name)
        images = next(iter(dataloader))[0][: self.log_k_images]
        captioned_images = _caption_images(images, trainer, pl_module)

        grid = make_grid(captioned_images, normalize=True)
        trainer.logger.experiment.add_image(
            f'{stage.capitalize()} predicts',
            img_tensor=grid,
            global_step=trainer.global_step,
        )


def _caption_images(images: torch.Tensor, trainer: Trainer, pl_module: OCRLightningModule) -> list[torch.Tensor]:
    raw_preds = pl_module(images.to(pl_module.device)).cpu().detach()
    string_preds, _ = matrix_to_string(raw_preds, trainer.datamodule.data_cfg.vocab)

    captioned_images = []
    for image, pred_text in zip(images, string_preds):
        captioned_img = denormalize(tensor_to_cv_image(image))
        captioned_img = cv_image_to_tensor(captioned_img)
        captioned_img = _caption_predict(captioned_img, pred_text)
        captioned_images.append(captioned_img)

    return captioned_images


# TODO: Split the function
def _caption_predict(image_tensor: torch.Tensor, text: str) -> torch.Tensor:  # noqa: WPS210
    image = to_pil_image(image_tensor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial.ttf', PRED_TEXT_SIZE)
    position = (10, 10)  # (x, y) coordinates for the upper left corner
    bbox = draw.textbbox(position, text, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    background_position = (position[0] - 5, position[1] - 5)
    background_size = (text_width + 10, text_height + 10)

    draw.rectangle(
        [
            background_position,
            (background_position[0] + background_size[0], background_position[1] + background_size[1]),
        ],
        fill=(0, 0, 0),
    )
    draw.text(position, text, font=font, fill=(255, 255, 255))  # Fill color is white
    return to_tensor(image)


class VisualizeBatch(VisualizeCallback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:  # noqa: WPS210
        self._log_batch(trainer, stage=TRAIN_STAGE)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:  # noqa: WPS210
        self._log_batch(trainer, stage='val')

    def on_test_epoch_start(self, trainer: Trainer, pl_module: OCRLightningModule) -> None:  # noqa: WPS210
        self._log_batch(trainer, stage='test')

    def _log_batch(self, trainer: Trainer, stage: str) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        dataloader_name = f'{stage}_dataloader' if stage == TRAIN_STAGE else f'{stage}_dataloaders'
        dataloader = getattr(trainer, dataloader_name)
        images = next(iter(dataloader))[0][: self.log_k_images]

        visualizations = []
        for img in images:
            img = denormalize(tensor_to_cv_image(img))
            visualizations.append(cv_image_to_tensor(img))

        grid = make_grid(images, normalize=True)
        trainer.logger.experiment.add_image(
            f'{stage.capitalize()} batch preview',
            img_tensor=grid,
            global_step=trainer.global_step,
        )
