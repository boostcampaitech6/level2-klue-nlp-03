from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassF1Score
from transformers import AutoConfig, AutoModel

from ..data.components.labelsets import label2id


class SemanticTypingModule(LightningModule):
    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        do_prob: float = 0.2,
        loss_fn: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        config = AutoConfig.from_pretrained(model_name)
        self.plm = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Sequential(
            nn.Dropout(do_prob),
            nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size),
            nn.GELU(),
            nn.Dropout(do_prob),
            nn.Linear(self.plm.config.hidden_size, 30),
        )

        # weight initialization
        self.classifier.apply(self._init_weights)

        # loss functions
        self.criterion = loss_fn

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
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, sent_input_ids, sent_attention_mask, desc_input_ids, desc_attention_mask):
        sent_outputs = self.plm(input_ids=sent_input_ids, attention_mask=sent_attention_mask)
        desc_outputs = self.plm(input_ids=desc_input_ids, attention_mask=desc_attention_mask)
        sent_poopled = sent_outputs.get("pooler_output", sent_outputs.last_hidden_state[:, 0])
        desc_poopled = desc_outputs.get("pooler_output", desc_outputs.last_hidden_state[:, 0])

        head_inputs = torch.cat((sent_poopled, desc_poopled), dim=-1)
        logits = self.classifier(head_inputs)
        return logits

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
        loss = self.criterion(logits, targets)
        return loss, logits, targets

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

    def on_test_start(self):
        if isinstance(self.logger, WandbLogger):
            self.test_logits = []
            self.test_targets = []

    def test_step(self, batch, batch_idx) -> None:
        loss, logits, targets = self.model_step(batch)
        if isinstance(self.logger, WandbLogger):
            self.test_logits.append(logits)
            self.test_targets.append(targets)

        # update and log metrics
        self.test_loss(loss)
        self.test_micro_f1(logits, targets)
        self.test_auprc(logits, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/micro_f1", self.test_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_end(self):
        if isinstance(self.logger, WandbLogger):
            named_labels = list(label2id.keys())
            logits = torch.cat(self.test_logits, dim=0)
            targets = torch.cat(self.test_targets, dim=0)

            probs = torch.nn.functional.softmax(logits, dim=1)

            targets_np, probs_np = targets.cpu().numpy(), probs.cpu().numpy()

            wandb.log(
                {
                    "precision_recall_curve": wandb.plot.pr_curve(
                        y_true=targets_np, y_probas=probs_np, labels=named_labels
                    )
                }
            )
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=probs_np, y_true=targets_np, class_names=named_labels
                    )
                }
            )

    def predict_step(self, batch, batch_idx):
        return self.forward(**batch)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_warmup_steps=0.1 * self.total_steps,
                num_training_steps=self.total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}

    def setup(self, stage=None):
        if stage == "fit":
            self.total_steps = self.trainer.max_epochs * len(
                self.trainer.datamodule.train_dataloader()
            )


if __name__ == "__main__":
    _ = SemanticTypingModule()
