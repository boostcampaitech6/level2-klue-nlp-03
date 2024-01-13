from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

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

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_prde: Optional[Dataset] = None

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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_pred,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = UniSTDataModule()
