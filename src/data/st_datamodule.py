from typing import Any, Dict, Optional

import os

import torch
from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .components.st_dataset import SemanticTypingDataset


class SemanticTypingDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        data_dir: str = "data/",
        file_train: str = "train.csv",
        file_val: str = "valid.csv",
        file_test: str = "test.csv",
        file_pred: str = "predict.csv",
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_pred: Optional[Dataset] = None

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_name)

    def collate_fn(self, batch):
        sentences = [sample["sentence"] for sample in batch]
        descriptions = [sample["description"] for sample in batch]
        sent_tokenized = self.tokenizer(sentences, truncation=True)
        desc_tokenized = self.tokenizer(descriptions, truncation=True)

        sent_collated = self.data_collator(sent_tokenized)
        desc_collated = self.data_collator(desc_tokenized)

        batch_preprocessed = {
            "sent_input_ids": sent_collated["input_ids"],
            "desc_input_ids": desc_collated["input_ids"],
            "sent_attention_mask": sent_collated["attention_mask"],
            "desc_attention_mask": desc_collated["attention_mask"],
        }

        if "labels" in batch[0].keys():
            labels = [sample["labels"] for sample in batch]
            batch_preprocessed["labels"] = torch.as_tensor(labels)

        return batch_preprocessed

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            ds_train = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"train": self.hparams.file_train},
                split="train",
            )
            ds_val = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"validation": self.hparams.file_val},
                split="validation",
            )

            self.data_train = SemanticTypingDataset(ds_train)
            self.data_val = SemanticTypingDataset(ds_val)

        elif stage == "test" or stage is None:
            ds_test = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"test": self.hparams.file_test},
                split="test",
            )
            self.data_test = SemanticTypingDataset(ds_test)

        elif stage == "predict":
            ds_pred = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"prediction": self.hparams.file_pred},
                split="prediction",
            )
            self.data_pred = SemanticTypingDataset(ds_pred, is_pred=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    _ = SemanticTypingDataModule()
