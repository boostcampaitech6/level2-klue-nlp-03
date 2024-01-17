from typing import Any, Dict, Optional

import os

import torch
from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .components.rbert_dataset import RBERTDataset


class RBERTDataModule(LightningDataModule):
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

        self.ssid = self.tokenizer.convert_tokens_to_ids("<SUBJ>")
        self.seid = self.tokenizer.convert_tokens_to_ids("</SUBJ>")
        self.osid = self.tokenizer.convert_tokens_to_ids("<OBJ>")
        self.oeid = self.tokenizer.convert_tokens_to_ids("</OBJ>")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_pred: Optional[Dataset] = None

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_name)

    def collate_fn(self, batch):
        sentences = [sample["sentence"] for sample in batch]
        descriptions = [sample["description"] for sample in batch]
        # print("*"*100, sentences[0])
        tokenized = self.tokenizer(sentences, descriptions, max_length=256, truncation=True)

        input_trimmed = [id[:-1] for id in tokenized["input_ids"]]
        token_type_trimmed = [tid[:-1] for tid in tokenized["token_type_ids"]]
        attention_trimmed = [am[:-1] for am in tokenized["attention_mask"]]

        max_length = max(len(am) for am in attention_trimmed)

        subject_mask = []
        object_mask = []

        for inp, am in zip(input_trimmed, attention_trimmed):
            try:
                ss, se, os, oe = (
                    inp.index(self.ssid),
                    inp.index(self.seid),
                    inp.index(self.osid),
                    inp.index(self.oeid),
                )
            except ValueError:
                # Handle the error if any of the IDs are not found
                continue

            length = len(am)
            sm = [1 if ss <= i <= se else 0 for i in range(length)]
            om = [1 if os <= i <= oe else 0 for i in range(length)]
            sm = sm + [0] * (max_length - len(sm))  # Padding
            om = om + [0] * (max_length - len(om))  # Padding

            subject_mask.append(sm)
            object_mask.append(om)

        batch_preprocessed = {
            "input_ids": input_trimmed,
            "token_type_ids": token_type_trimmed,
            "attention_mask": attention_trimmed,
            "subject_mask": subject_mask,
            "object_mask": object_mask,
        }

        if "labels" in batch[0].keys():
            labels = [sample["labels"] for sample in batch]
            batch_preprocessed["labels"] = torch.as_tensor(labels)

        return self.data_collator(batch_preprocessed)

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

            self.data_train = RBERTDataset(ds_train)
            self.data_val = RBERTDataset(ds_val)

        elif stage == "test" or stage is None:
            ds_test = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"test": self.hparams.file_test},
                split="test",
            )
            self.data_test = RBERTDataset(ds_test)

        elif stage == "predict":
            ds_pred = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files={"prediction": self.hparams.file_pred},
                split="prediction",
            )
            self.data_pred = RBERTDataset(ds_pred, is_pred=True)

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
    _ = RBERTDataModule()
