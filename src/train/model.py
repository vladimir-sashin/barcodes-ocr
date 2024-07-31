import torch
from timm import create_model
from torch import nn

from src.train.config.lightning_module_cfg import BackboneConfig, RNNConfig
from src.train.train_utils.serialization import load_object


class CRNN(nn.Module):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        rnn_cfg: RNNConfig,
    ) -> None:
        super().__init__()

        # Pretrained feature extraction CNN backbone
        self.backbone = create_model(
            backbone_cfg.backbone_name,
            pretrained=backbone_cfg.pretrained,
            features_only=True,
            output_stride=backbone_cfg.output_stride,
            out_indices=backbone_cfg.out_indices,
        )

        self.gate = nn.Conv2d(backbone_cfg.cnn_output_size, rnn_cfg.features_num, kernel_size=1, bias=False)

        # RNN
        self.rnn = load_object(rnn_cfg.target_model_class)(
            input_size=rnn_cfg.input_size,
            hidden_size=rnn_cfg.hidden_size,
            dropout=rnn_cfg.dropout,
            bidirectional=rnn_cfg.bidirectional,
            num_layers=rnn_cfg.num_layers,
        )

        classifier_in_features = rnn_cfg.hidden_size
        if rnn_cfg.bidirectional:
            classifier_in_features = 2 * rnn_cfg.hidden_size

        # Classificator
        self.fc = nn.Linear(classifier_in_features, rnn_cfg.num_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        cnn_features = self.backbone(tensor)[0]  # Feature map: BS x C_fm x H_fm x W_fm
        # Conv1x1 to set C_fm = rnn_features_num
        cnn_features = self.gate(cnn_features)  # BS x rnn_features_num x H_fm x W_fm
        # Permute for RNN (batch_first=False)
        cnn_features = cnn_features.permute(3, 0, 2, 1)  # W_fm x BS x  H_fm x rnn_features_num
        cnn_features = cnn_features.reshape(  # W_fm x BS x rnn_input_size
            cnn_features.shape[0],  # W_fm (number of slices)
            cnn_features.shape[1],  # BS
            cnn_features.shape[2] * cnn_features.shape[3],  # Slices' length = rnn_input_size = H_fm * rnn_features_num
        )
        rnn_output, _ = self.rnn(cnn_features)  # W_fm x BS x hidden_size (hidden_size * 2 if bidirectional)
        logits = self.fc(rnn_output)  # W_fm x BS x num_classes
        return self.softmax(logits)
