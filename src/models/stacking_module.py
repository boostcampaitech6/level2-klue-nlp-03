from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassF1Score


class StackingModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module = None,
        binary_loss_fn: torch.nn.Module = None,
        multiclass_loss_fn: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.binary_head = torch.nn.Linear(self.net.config.hidden_size, 2)
        self.multiclass_head = torch.nn.Linear(self.net.config.hidden_size, 30)

        # weight initialization
        self._init_weights(self.binary_head)
        self._init_weights(self.multiclass_head)

        # loss functions
        self.binary_criterion = binary_loss_fn
        self.multiclass_criterion = multiclass_loss_fn

        # loss sclaes for combining two losses
        self.loss_scales = torch.nn.Parameter(torch.ones(2))

        # metrics
        self.train_micro_f1 = (
            MulticlassF1Score(num_classes=30, average="micro", ignore_index=0) * 100
        )
        self.val_micro_f1 = (
            MulticlassF1Score(num_classes=30, average="micro", ignore_index=0) * 100
        )
        self.test_micro_f1 = (
            MulticlassF1Score(num_classes=30, average="micro", ignore_index=0) * 100
        )

        self.train_auprc = (
            MulticlassAveragePrecision(num_classes=30, average="macro", ignore_index=0) * 100
        )
        self.val_auprc = (
            MulticlassAveragePrecision(num_classes=30, average="macro", ignore_index=0) * 100
        )
        self.test_auprc = (
            MulticlassAveragePrecision(num_classes=30, average="macro", ignore_index=0) * 100
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation metrics (micro_f1, auprc)
        self.val_micro_f1_best = MaxMetric()
        self.val_auprc_best = MaxMetric()

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, **inputs):
        return self.net(**inputs)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_micro_f1.reset()
        self.val_micro_f1_best.reset()
        self.val_auprc.reset()
        self.val_auprc_best.reset()

    def model_step(self, batch):
        inputs = {key: val for key, val in batch.items() if key != "labels"}
        targets = batch["labels"]
        logits = self.forward(**inputs)

        # if the label is 'no_relation', it would be 1 else 0
        is_no_relation = torch.where(targets == 0, 1, 0)

        # binary classification phase
        binary_logits = self.binary_head(logits)
        binary_loss = self.binary_criterion(binary_logits, is_no_relation)

        # multiclass classification phase
        multiclass_logits = self.multiclass_head(logits)
        multiclass_loss = self.multiclass_criterion(multiclass_logits, targets)

        # combining two losses
        scales = torch.softmax(self.loss_scales, dim=0)
        total_loss = scales[0] * binary_loss + scales[1] * multiclass_loss

        return total_loss, multiclass_logits, targets

    def training_step(self, batch, batch_idx):
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_micro_f1(logits, targets)
        self.train_auprc(logits, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/micro_f1", self.train_micro_f1, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_micro_f1(logits, targets)
        self.val_auprc(logits, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/micro_f1", self.val_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        micro_f1 = self.val_micro_f1.compute()  # get current val micro f1
        self.val_micro_f1_best(micro_f1)  # update best so far val micro f1
        auprc = self.val_auprc.compute()  # get current val micro f1
        self.val_auprc_best(auprc)  # update best so far val micro f1

        # log `val_micro_f1_best` and 'val_auprc_best' as values through `.compute()` method, instead of as a metric object
        # otherwise metrics would be reset by lightning after each epoch
        self.log(
            "val/micro_f1_best", self.val_micro_f1_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log("val/auprc_best", self.val_auprc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_micro_f1(logits, targets)
        self.test_auprc(logits, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/micro_f1", self.test_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inputs = {key: val for key, val in batch.items() if key != "labels"}
        logits = self.forward(**inputs)
        return self.multiclass_head(logits)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = StackingModule()