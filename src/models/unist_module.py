from typing import Any, Dict, Tuple

import os

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassF1Score
from transformers import AutoTokenizer

from ..data.components.labelsets import labelset_ko


class UniSTModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        model_name = self.net.model.config._name_or_path
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens_dict = {
            "additional_special_tokens": ["<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.labelset = labelset_ko
        self.labelset_inputs = self.tokenizer(
            self.labelset, padding=True, truncation=True, return_tensors="pt"
        )

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

    def forward(self, inputs):
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
        inputs = {}
        texts_inputs = self.tokenizer(
            batch["sentence"],
            batch["description"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels_inputs = self.tokenizer(
            batch["labels"], padding=True, truncation=True, return_tensors="pt"
        )
        false_inputs = self.tokenizer(
            batch["false"], padding=True, truncation=True, return_tensors="pt"
        )

        inputs["texts_inputs"] = texts_inputs
        inputs["labels_inputs"] = labels_inputs
        inputs["false_inputs"] = false_inputs

        targets = batch["labels"]
        loss, embeddings = self.forward(inputs)
        return loss, embeddings, targets

    def training_step(self, batch, batch_idx):
        loss, embeddings, targets = self.model_step(batch)
        target_ids = torch.as_tensor([self.labelset.index(target) for target in targets]).to(
            "cuda:0"
        )

        labelset_embeddings = self.net.embed(**self.labelset_inputs)

        dists = []
        for i in range(len(embeddings)):
            dist = self.net.dist_fn(embeddings[i], labelset_embeddings)
            dists.append(dist)
        logits = 1 - torch.stack(dists)

        # update and log metrics
        self.train_loss(loss)
        self.train_micro_f1(logits, target_ids)
        self.train_auprc(logits, target_ids)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/micro_f1", self.train_micro_f1, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss, embeddings, targets = self.model_step(batch)
        target_ids = torch.as_tensor([self.labelset.index(target) for target in targets]).to(
            "cuda:0"
        )

        labelset_embeddings = self.net.embed(**self.labelset_inputs)

        dists = []
        for i in range(len(embeddings)):
            dist = self.net.dist_fn(embeddings[i], labelset_embeddings)
            dists.append(dist)
        logits = 1.0 - torch.stack(dists)

        # update and log metrics
        self.val_loss(loss)
        self.val_micro_f1(logits, target_ids)
        self.val_auprc(logits, target_ids)
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
        loss, embeddings, targets = self.model_step(batch)
        target_ids = torch.as_tensor([self.labelset.index(target) for target in targets]).to(
            "cuda:0"
        )

        labelset_embeddings = self.net.embed(**self.labelset_inputs)

        dists = []
        for i in range(len(embeddings)):
            dist = self.net.dist_fn(embeddings[i], labelset_embeddings)
            dists.append(dist)
        logits = 1 - torch.stack(dists)

        # update and log metrics
        self.test_loss(loss)
        self.test_micro_f1(logits, target_ids)
        self.test_auprc(logits, target_ids)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/micro_f1", self.test_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inputs = self.tokenizer(
            batch["sentence"],
            batch["description"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embeddings = self.net.embed(**inputs)
        labelset_embeddings = self.net.embed(**inputs)
        dists = []
        for i in range(len(embeddings)):
            dist = self.net.dist_fn(embeddings[i], labelset_embeddings)
            dists.append(dist)
        logits = 1.0 - torch.stack(dists)
        return logits

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
    _ = UniSTModule()
