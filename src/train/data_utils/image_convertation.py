import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor

from src.train.data_utils.constants import IMAGE_KEY


def denormalize(
    img: NDArray[float],
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
) -> NDArray[int]:
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)[IMAGE_KEY] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    return tensor.permute(1, 2, 0).cpu().numpy()


def cv_image_to_tensor(image: NDArray[float]) -> Tensor:
    transform = ToTensorV2()
    return transform(image=image)[IMAGE_KEY]
