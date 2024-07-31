from typing import Any

import lightning
import torch
from torch import Tensor

from src.constants import MONITOR_METRIC
from src.train.config.lightning_module_cfg import LightningModuleConfig
from src.train.model import CRNN
from src.train.train_utils.losses import get_losses
from src.train.train_utils.metrics.metric_collection import get_metrics
from src.train.train_utils.serialization import load_object


class OCRLightningModule(lightning.LightningModule):  # noqa: WPS214 # Required by Lightning
    def __init__(self, config: LightningModuleConfig):
        super().__init__()
        self.cfg = config

        self.model = CRNN(self.cfg.backbone_cfg, self.cfg.rnn_cfg)

        self._losses = get_losses(self.cfg.losses)

        metrics = get_metrics()
        self._train_metrics = metrics.clone(postfix='/train')
        self._valid_metrics = metrics.clone(postfix='/valid')
        self._test_metrics = metrics.clone(postfix='/test')

        self.save_hyperparameters()

        self.best_epoch = 0
        self.best_metric = float('-inf')

    def forward(self, batch: Tensor) -> Tensor:
        return self.model(batch)

    def configure_optimizers(self) -> dict[Any]:  # type: ignore # Allow explicit Any
        optimizer = load_object(self.cfg.optimizer.target_class)(
            self.model.parameters(),
            **self.cfg.optimizer.kwargs,
        )
        scheduler_cls = load_object(self.cfg.scheduler.target_class)
        scheduler = scheduler_cls(optimizer, **self.cfg.scheduler.kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                **self.cfg.scheduler.lightning_kwargs,
            },
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss_value, log_probs, targets = self._common_step(batch, 'train')

        self._train_metrics(log_probs, targets)

        return loss_value

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        loss_value, log_probs, targets = self._common_step(batch, 'valid')

        self._valid_metrics(log_probs, targets)

    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        images, targets, target_lengths = batch
        log_probs = self(images)
        self._test_metrics(log_probs, targets)

    def on_train_epoch_start(self) -> None:
        self._train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.log_dict(self._train_metrics.compute(), on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True, prog_bar=True, logger=True)
        self._log_best_epoch_num()

    def on_test_epoch_start(self) -> None:
        self._test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True, prog_bar=True, logger=True)

    def _common_step(self, batch: Tensor, stage: str) -> tuple[Tensor, Tensor, Tensor]:
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = [log_probs.size(0) for _ in range(images.size(0))]
        input_lengths = torch.IntTensor(input_lengths)
        loss_value = self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            stage,
        )

        return loss_value, log_probs, targets

    def _log_best_epoch_num(self) -> None:
        current_epoch = self.current_epoch
        current_metric = self.trainer.callback_metrics[MONITOR_METRIC]
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = current_epoch
        self.log('best_epoch', self.best_epoch, on_epoch=True, prog_bar=True, logger=True)

    def _calculate_loss(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        postfix: str,
    ) -> Tensor:
        total_loss = torch.tensor(0, dtype=torch.float32, device=log_probs.device)
        for cur_loss in self._losses:
            loss = cur_loss.loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            total_loss += cur_loss.weight * loss
            self.log(
                f'{cur_loss.name}_loss/{postfix}',
                loss.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log(
            name=f'total_loss/{postfix}',
            value=total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss
