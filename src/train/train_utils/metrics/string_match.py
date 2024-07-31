import itertools

import numpy as np
import torch
from torchmetrics import Metric

from src.train.train_utils.metrics.metrics_utils import (
    DIST_REDUCE_FX,
    prep_batches,
)


class StringMatchMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            'correct',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=DIST_REDUCE_FX,
        )
        self.add_state(
            'total',
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx=DIST_REDUCE_FX,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        batch_size = torch.tensor(target.shape[0])

        metric = torch.tensor(string_match(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        return self.correct / self.total


def string_match(preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> float:
    preds_batch, gt_batch = prep_batches(preds_batch, gt_batch)
    return _calculate_string_match(preds_batch, gt_batch)


def _calculate_string_match(preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> float:
    str_match = float(0)
    for idx in range(preds_batch.shape[0]):
        pred_chars = [char for char, _ in itertools.groupby(preds_batch[idx])]
        pred_chars = [char for char in pred_chars if char > 0]
        gt_chars = [char for char in gt_batch[idx] if char > 0]
        str_match += float(np.array_equal(pred_chars, gt_chars))

    return str_match / preds_batch.shape[0]
