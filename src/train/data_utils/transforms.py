from typing import Any, Union

import albumentations as albu
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numpy import random
from numpy.typing import NDArray

from src.train.config.datamodule_cfg import DataConfig
from src.train.data_utils.constants import (  # noqa: WPS235 # TODO: Parametrize in config
    BLUR_LIMIT,
    BLUR_P,
    BRIGHT_CONTRAST_P,
    CLAHE_P,
    CROP_PERSPECTIVE_P,
    DOWNSCALE_MAX,
    DOWNSCALE_MIN,
    DOWNSCALE_P,
    DROPOUT_MAX_HOLES,
    DROPOUT_MIN_HOLES,
    DROPOUT_P,
    GAUSS_P,
    IMAGE_KEY,
    RANDOM_MODE,
    SCALE_X_P,
)
from src.train.data_utils.types import TRANSFORM_TYPE


def get_transforms(
    data_cfg: DataConfig,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    transforms = []

    if augmentations:
        transforms.extend(
            [
                CropPerspective(p=CROP_PERSPECTIVE_P),
                ScaleX(p=SCALE_X_P),
            ],
        )

    if preprocessing:
        transforms.append(
            PadResizeOCR(
                target_height=data_cfg.height,
                target_width=data_cfg.width,
                mode=RANDOM_MODE if augmentations else 'left',
            ),
        )

    if augmentations:
        transforms.extend(
            [
                albu.RandomBrightnessContrast(p=BRIGHT_CONTRAST_P),
                albu.CLAHE(p=CLAHE_P),
                albu.Blur(blur_limit=BLUR_LIMIT, p=BLUR_P),
                albu.GaussNoise(p=GAUSS_P),
                albu.Downscale(scale_min=DOWNSCALE_MIN, scale_max=DOWNSCALE_MAX, p=DOWNSCALE_P),
                albu.CoarseDropout(min_holes=DROPOUT_MIN_HOLES, max_holes=DROPOUT_MAX_HOLES, p=DROPOUT_P),
            ],
        )

    if postprocessing:
        transforms.extend(
            [
                albu.Normalize(),
                TextEncode(vocab=data_cfg.vocab, target_text_size=data_cfg.text_size),
                ToTensorV2(),
            ],
        )

    return albu.Compose(transforms)


class PadResizeOCR:
    """Resize keeping initial aspect ratio using padding (letterbox resize)."""

    def __init__(self, target_width: int, target_height: int, mode: str = RANDOM_MODE):
        self.target_width = target_width
        self.target_height = target_height
        self.mode = mode

        if self.mode not in {RANDOM_MODE, 'left', 'center'}:
            raise ValueError(f'`mode` must be one of {{{RANDOM_MODE}, left, center}}, got {self.mode}.')

    def __call__(self, **kwargs: Any) -> dict[str, NDArray]:  # type: ignore # Allow explicit Any
        image = kwargs[IMAGE_KEY].copy()
        image, tmp_w = self._resize_image(image)
        image = self._pad_image(image, tmp_w)
        kwargs[IMAGE_KEY] = image
        return kwargs

    def _resize_image(self, image: NDArray) -> tuple[NDArray, int]:
        height, width = image.shape[:2]

        tmp_w = min(int(width * (self.target_height / height)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))
        return image, tmp_w

    def _pad_image(self, image: NDArray, tmp_w: int) -> NDArray:
        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == RANDOM_MODE:
                pad_left = np.random.randint(dw)
            elif self.mode == 'left':
                pad_left = 0
            else:
                pad_left = dw // 2

            pad_right = dw - pad_left

            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image


class TextEncode:
    def __init__(self, vocab: Union[str, list[str]], target_text_size: int):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size = target_text_size

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore # Allow explicit Any
        source_text = kwargs['text'].strip()

        # TODO: replace by for loop for readability
        postprocessed_text = [self.vocab.index(char) + 1 for char in source_text if char in self.vocab]
        postprocessed_text = np.pad(
            postprocessed_text,
            pad_width=(0, self.target_text_size - len(postprocessed_text)),
            mode='constant',
        )

        kwargs['text'] = torch.IntTensor(postprocessed_text)

        return kwargs


class CropPerspective:
    def __init__(
        self,
        p: float = 0.5,  # noqa: WPS111 albu style
        width_ratio: float = 0.04,
        height_ratio: float = 0,
    ):
        self.p = p  # noqa: WPS111 albu style
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def __call__(self, **kwargs: Any) -> dict[str, NDArray]:  # type: ignore # Allow explicit Any
        image = kwargs[IMAGE_KEY].copy()

        if random.random() < self.p:
            height, width, ch = image.shape

            pts1 = np.float32(
                [
                    [0, 0],
                    [0, height],
                    [width, height],
                    [width, 0],
                ],
            )
            pts2 = self._get_pts2(height, width)

            image = _transform_image(image, pts1, pts2)

        kwargs[IMAGE_KEY] = image
        return kwargs

    def _get_pts2(self, height: int, width: int) -> NDArray:
        dh = height * self.height_ratio
        dw = width * self.width_ratio
        return np.float32(
            [
                [random.uniform(-dw, dw), random.uniform(-dh, dh)],
                [random.uniform(-dw, dw), height - random.uniform(-dh, dh)],
                [width - random.uniform(-dw, dw), height - random.uniform(-dh, dh)],
                [width - random.uniform(-dw, dw), random.uniform(-dh, dh)],
            ],
        )


def _transform_image(image: NDArray, pts1: NDArray, pts2: NDArray) -> NDArray:
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    dst_w = (pts2[3][0] + pts2[2][0] - pts2[1][0] - pts2[0][0]) * 0.5
    dst_h = (pts2[2][1] + pts2[1][1] - pts2[3][1] - pts2[0][1]) * 0.5
    return cv2.warpPerspective(
        image,
        matrix,
        dsize=(int(dst_w), int(dst_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class ScaleX:
    def __init__(self, p: float = 0.5, scale_min: float = 0.8, scale_max: float = 1.2):  # noqa: WPS111 albu style
        self.p = p  # noqa: WPS111 albu style
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, **kwargs: Any) -> dict[str, NDArray]:  # type: ignore # Allow explicit Any
        image = kwargs[IMAGE_KEY].copy()

        if random.random() < self.p:
            height, width, ch = image.shape
            width = int(width * random.uniform(self.scale_min, self.scale_max))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        kwargs[IMAGE_KEY] = image
        return kwargs
