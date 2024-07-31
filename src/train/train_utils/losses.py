from dataclasses import dataclass

from torch import nn

from src.train.config.lightning_module_cfg import LossConfig
from src.train.train_utils.serialization import load_object


@dataclass
class Loss:
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: list[LossConfig]) -> list[Loss]:
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.target_class)(**loss_cfg.kwargs),
        )
        for loss_cfg in losses_cfg
    ]
