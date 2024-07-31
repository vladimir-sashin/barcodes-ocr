import torch

DIST_REDUCE_FX = 'sum'


def prep_batches(preds_batch: torch.Tensor, gt_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preds_batch = preds_batch.permute(1, 0, 2)
    preds_batch = torch.Tensor.argmax(preds_batch, dim=2)
    preds_batch = preds_batch.detach().cpu().numpy()

    gt_batch = gt_batch.detach().cpu().numpy()

    return preds_batch, gt_batch
