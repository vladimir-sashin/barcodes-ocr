import itertools

import torch
from nltk import edit_distance as nltk_edit_dist
from torchmetrics import Metric

from src.train.train_utils.metrics.metrics_utils import (
    DIST_REDUCE_FX,
    prep_batches,
)


class EditDistanceMetric(Metric):
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

        metric = torch.tensor(edit_distance(preds, target))

        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        return self.correct / self.total


def edit_distance(preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> float:
    preds_batch, gt_batch = prep_batches(preds_batch, gt_batch)
    return _calculate_edit_dist(preds_batch, gt_batch)


def _calculate_edit_dist(preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> float:
    edit_dist = 0

    for idx in range(preds_batch.shape[0]):
        str_pred, str_true = _calculate_single_dist(idx, preds_batch, gt_batch)
        edit_dist += nltk_edit_dist(str_pred, str_true)

    return edit_dist / preds_batch.shape[0]


def _calculate_single_dist(idx: int, preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> tuple[str, str]:
    pred_chars = [char for char, _ in itertools.groupby(preds_batch[idx])]
    pred_chars = [char for char in pred_chars if char > 0]
    gt_chars = [char for char in gt_batch[idx] if char > 0]

    str_pred = ''.join([chr(one_pred) for one_pred in pred_chars])
    str_gt = ''.join([chr(one_pred) for one_pred in gt_chars])

    return str_pred, str_gt
