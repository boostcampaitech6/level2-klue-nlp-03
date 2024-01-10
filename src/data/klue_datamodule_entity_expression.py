from typing import Any, Dict, Optional

import pickle
from ast import literal_eval

from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class KLUEDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        data_dir: str = "data/",
        file_train: str = "train.csv",
        file_val: str = "valid.csv",
        file_test: str = "test.csv",
        file_pred: str = "predict.csv",
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        self.id2label = None
        self.label2id = None

    @property
    def num_classes(self) -> int:
        return 30

    def load_dataset_file(self, file_type: str, stage=None) -> Dataset:
        try:
            dataset = load_dataset(
                "csv",
                data_dir=self.hparams.data_dir,
                data_files=getattr(self.hparams, f"file_{file_type}"),
                split="train",
            )
            return self.preprocessing(dataset, stage)
        except Exception as e:
            print(f"Error loading dataset file {file_type}: {e}")
            return None

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_name)

    def encode_label_to_id(self, examples):
        examples["label"] = [self.label2id[example] for example in examples["label"]]
        return examples

    def get_word_word_type_and_idxs_from_entity(self, examples):
        return {
            "subject_entity_type": [
                literal_eval(example)["type"] for example in examples["subject_entity"]
            ],
            "object_entity_type": [
                literal_eval(example)["type"] for example in examples["object_entity"]
            ],
            "subject_entity_start_idx": [
                literal_eval(example)["start_idx"] for example in examples["subject_entity"]
            ],
            "subject_entity_end_idx": [
                literal_eval(example)["end_idx"] for example in examples["subject_entity"]
            ],
            "object_entity_start_idx": [
                literal_eval(example)["start_idx"] for example in examples["object_entity"]
            ],
            "object_entity_end_idx": [
                literal_eval(example)["end_idx"] for example in examples["object_entity"]
            ],
        }

    def entity_expression(self, examples):
        ko_entity_type = {
            "PER": "사람",
            "ORG": "단체",
            "DAT": "날짜",
            "LOC": "장소",
            "POH": "본사",
            "NOH": "수량",
        }
        return {
            "text": [
                s[:sesi]
                + "@ * "
                + ko_entity_type[set]
                + " * "
                + s[sesi : seei + 1]
                + " @ "
                + s[seei + 1 : oesi]
                + " # ^ "
                + ko_entity_type[oet]
                + " ^ "
                + s[oesi : oeei + 1]
                + " # "
                + s[oeei + 1 :]
                if sesi < oesi
                else s[:oesi]
                + "# ^ "
                + ko_entity_type[oet]
                + " ^ "
                + s[oesi : oeei + 1]
                + " # "
                + s[oeei + 1 : sesi]
                + "@ * "
                + ko_entity_type[set]
                + " * "
                + s[sesi : seei + 1]
                + " @ "
                + s[seei + 1 :]
                for s, set, oet, sesi, seei, oesi, oeei in zip(
                    examples["sentence"],
                    examples["subject_entity_type"],
                    examples["object_entity_type"],
                    examples["subject_entity_start_idx"],
                    examples["subject_entity_end_idx"],
                    examples["object_entity_start_idx"],
                    examples["object_entity_end_idx"],
                )
            ]
        }

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def preprocessing(self, dataset, stage):
        dataset = dataset.map(self.get_word_word_type_and_idxs_from_entity, batched=True)
        dataset = dataset.map(self.entity_expression, batched=True)
        dataset = dataset.map(self.tokenize_function, batched=True)
        dataset = dataset.select_columns(
            ["input_ids", "token_type_ids", "attention_mask", "label"]
        )
        if stage != "predict":
            dataset = dataset.map(self.encode_label_to_id, batched=True)
        else:
            dataset = dataset.remove_columns("label")
        return dataset.with_format("torch")

    def setup(self, stage: Optional[str] = None) -> None:
        try:
            with open(self.hparams.data_dir + "label2id.pkl", "rb") as file:
                self.label2id = pickle.load(file)
        except FileNotFoundError:
            print("Error: Label mapping files not found.")

        if stage == "fit" or stage is None:
            self.data_train = self.load_dataset_file("train", stage)
            self.data_val = self.load_dataset_file("val", stage)

        elif stage == "test" or stage is None:
            self.data_test = self.load_dataset_file("test", stage)

        elif stage == "predict":
            self.data_pred = self.load_dataset_file("pred", stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = KLUEDataModule()
