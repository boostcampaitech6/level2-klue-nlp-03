from typing import Any, Dict, Tuple

import os

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassF1Score
from transformers import AutoConfig, AutoModel

from ..data.components.labelsets import labelset_ko


class UniSTModule(LightningModule):
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        margin: float = 0.1,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        config = AutoConfig.from_pretrained(model_name)
        self.net = AutoModel.from_pretrained(model_name, config=config)

        self.margin = margin
        self.labelset_input_ids = None
        self.labelset_attention_mask = None

        self.net.apply(self.init_weights)

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

        self.train_step_embeddings = []
        self.train_step_labels = []

        self.validation_step_embeddings = []
        self.validation_step_labels = []

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def convert_to_logits(self, dists):
        dists = torch.stack(dists)
        eps = 1e-6
        clipped_tensor = torch.clamp(dists, 1 - eps)
        logits = torch.log(clipped_tensor / (1 - clipped_tensor))
        return logits

    def forward(
        self,
        texts_input_ids,
        labels_input_ids,
        fake_input_ids,
        texts_attention_mask=None,
        labels_attention_mask=None,
        fake_attention_mask=None,
    ):
        texts_embeddings = self.embed(texts_input_ids, texts_attention_mask)
        labels_embeddings = self.embed(labels_input_ids, labels_attention_mask)
        fake_embeddings = self.embed(fake_input_ids, fake_attention_mask)
        loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: F.cosine_similarity(x, y), margin=self.margin
        )

        loss = loss_fn(texts_embeddings, fake_embeddings, labels_embeddings)

        return loss, texts_embeddings

    def embed(
        self,
        input_ids,
        attention_mask=None,
    ):
        outputs = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_outputs = outputs.get("pooler_output", outputs.last_hidden_state[:, 0])

        return pooled_outputs

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_micro_f1.reset()
        self.val_micro_f1_best.reset()
        self.val_auprc.reset()
        self.val_auprc_best.reset()

    def model_step(self, batch):
        inputs = {key: val for key, val in batch.items() if key != "label_ids"}
        label_ids = torch.as_tensor(batch["label_ids"])

        loss, embeddings = self.forward(**inputs)
        return loss, embeddings, label_ids

    def training_step(self, batch, batch_idx):
        loss, embeddings, label_ids = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_step_embeddings.append(embeddings)
        self.train_step_labels.append(label_ids)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        embeddings = torch.cat(self.train_step_embeddings, dim=0)
        label_ids = torch.cat(self.train_step_labels, dim=0)

        with torch.no_grad():
            labelset_embeddings = self.embed(
                self.labelset_input_ids.to(self.device),
                self.labelset_attention_mask.to(self.device),
            )

            dists = []
            for i in range(len(embeddings)):
                embedding = embeddings[i].expand(labelset_embeddings.shape)
                dist = F.cosine_similarity(embedding, labelset_embeddings).detach().cpu()
                dists.append(dist)

            logits = self.convert_to_logits(dists)

        self.train_micro_f1(logits, label_ids)
        self.train_auprc(logits, label_ids)
        self.log(
            "train/micro_f1", self.train_micro_f1, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True)

        self.train_step_embeddings.clear()
        self.train_step_labels.clear()

    def validation_step(self, batch, batch_idx) -> None:
        loss, embeddings, label_ids = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_embeddings.append(embeddings)
        self.validation_step_labels.append(label_ids)

    def on_validation_epoch_end(self):
        embeddings = torch.cat(self.validation_step_embeddings, dim=0)
        label_ids = torch.cat(self.validation_step_labels, dim=0)
        with torch.no_grad():
            labelset_embeddings = self.embed(
                self.labelset_input_ids.to(self.device),
                self.labelset_attention_mask.to(self.device),
            )

            dists = []
            for i in range(len(embeddings)):
                embedding = embeddings[i].expand(labelset_embeddings.shape)
                dist = F.cosine_similarity(embedding, labelset_embeddings).detach().cpu()
                dists.append(dist)

            logits = self.convert_to_logits(dists)

        self.val_micro_f1(logits, label_ids)
        self.val_auprc(logits, label_ids)
        self.log("val/micro_f1", self.val_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)

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
        self.validation_step_embeddings.clear()
        self.validation_step_labels.clear()

    def test_step(self, batch, batch_idx) -> None:
        loss, embeddings, label_ids = self.model_step(batch)

        with torch.no_grad():
            labelset_embeddings = self.embed(
                self.labelset_input_ids.to(self.device),
                self.labelset_attention_mask.to(self.device),
            )

            dists = []
            for i in range(len(embeddings)):
                embedding = embeddings[i].expand(labelset_embeddings.shape)
                dist = F.cosine_similarity(embedding, labelset_embeddings).detach().cpu()
                dists.append(dist)

            logits = self.convert_to_logits(dists)

        # update and log metrics
        self.test_loss(loss)
        self.test_micro_f1(logits, label_ids)
        self.test_auprc(logits, label_ids)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/micro_f1", self.test_micro_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        input_ids = batch["texts_input_ids"]
        attention_mask = batch["texts_attention_mask"]

        embeddings = self.embed(input_ids, attention_mask)
        with torch.no_grad():
            labelset_embeddings = self.embed(self.labelset_input_ids, self.labelset_attention_mask)

            dists = []
            for i in range(len(embeddings)):
                embedding = embeddings[i].expand(labelset_embeddings.shape)
                dist = F.cosine_similarity(embedding, labelset_embeddings).detach().cpu()
                dists.append(dist)

            logits = self.convert_to_logits(dists)

        return logits

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
    _ = UniSTModule()