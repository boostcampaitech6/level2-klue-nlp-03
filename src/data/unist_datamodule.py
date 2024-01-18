from typing import Any, Dict, Optional

import os

import torch
from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .components.labelsets import labelset_ko
from .components.unist_dataset import UniSTDataset


class UniSTDataModule(LightningDataModule):
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
        special_tokens_dict = {
            "additional_special_tokens": ["<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

        tokenized_labelset = self.tokenizer(labelset_ko, truncation=True, max_length=13)
        collated_labelset = self.data_collator(tokenized_labelset)
        self.labelset_input_ids = collated_labelset["input_ids"]
        self.labelset_attention_mask = collated_labelset["attention_mask"]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_pred: Optional[Dataset] = None

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_name)

    def collate_fn(self, batch):
        sentences = [sample["sentence"] for sample in batch]
        descriptions = [sample["description"] for sample in batch]

        tokenized_texts = self.tokenizer(sentences, descriptions, max_length=256, truncation=True)
        collated_texts = self.data_collator(tokenized_texts)

        if "labels" in batch[0].keys():
            labels = [sample["labels"] for sample in batch]
            fake = [sample["fake"] for sample in batch]

            tokenized_labels = self.tokenizer(labels, max_length=13, truncation=True)
            collated_labels = self.data_collator(tokenized_labels)

            tokenized_fake = self.tokenizer(fake, max_length=13, truncation=True)
            collated_fake = self.data_collator(tokenized_fake)

            label_ids = [sample["label_ids"] for sample in batch]

            return {
                "texts_input_ids": collated_texts["input_ids"],
                "labels_input_ids": collated_labels["input_ids"],
                "fake_input_ids": collated_fake["input_ids"],
                "texts_attention_mask": collated_texts["attention_mask"],
                "labels_attention_mask": collated_labels["attention_mask"],
                "fake_attention_mask": collated_fake["attention_mask"],
                "label_ids": torch.as_tensor(label_ids),
            }
        else:
            return {
                "texts_input_ids": collated_texts["input_ids"],
                "texts_attention_mask": collated_texts["attention_mask"],
            }

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

            self.data_train = UniSTDataset(ds_train)
            self.data_val = UniSTDataset(ds_val)

        elif stage == "test" or stage is None:
            ds_test = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"test": self.hparams.file_test},
                split="test",
            )
            self.data_test = UniSTDataset(ds_test)

        elif stage == "predict":
            ds_pred = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"prediction": self.hparams.file_pred},
                split="prediction",
            )
            self.data_pred = UniSTDataset(ds_pred, is_pred=True)

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
    _ = UniSTDataModule()