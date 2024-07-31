from typing import Any, Optional

from pydantic import Field

from src.base_config import BaseValidatedConfig


class BackboneConfig(BaseValidatedConfig):
    backbone_name: str = 'resnet18'
    pretrained: bool = True
    cnn_output_size: int = 128
    output_stride: Optional[int] = None
    out_indices: Optional[tuple[int]] = (2,)


class RNNConfig(BaseValidatedConfig):
    target_model_class: str = 'torch.nn.GRU'
    input_size: int = 576
    features_num: int = 48
    hidden_size: int = 64
    dropout: float = 0.1
    bidirectional: bool = True
    num_layers: int = 2
    num_classes: int = 11


class SerializableObject(BaseValidatedConfig):
    target_class: str
    kwargs: dict[str, Any] = Field(default_factory=dict)  # type: ignore  # Allow explicit `Any`


class LossConfig(SerializableObject):
    name: str
    weight: float


class SchedulerConfig(SerializableObject):
    lightning_kwargs: dict[str, Any] = Field(default_factory=dict)  # type: ignore  # Allow explicit `Any`


class LightningModuleConfig(BaseValidatedConfig):
    backbone_cfg: BackboneConfig
    rnn_cfg: RNNConfig
    optimizer: SerializableObject
    scheduler: SchedulerConfig
    losses: list[LossConfig]
