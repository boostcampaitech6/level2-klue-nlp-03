from pathlib import Path

import pytest
import torch

from src.data.klue_datamodule import KLUEDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    model_name: str = "klue/bert-base"
    data_dir: str = "./data/"
    file_train: str = "train.csv"
    file_val: str = "val.csv"
    file_test: str = "test.csv"
    file_pred: str = "pred.csv"
    batch_size: int = 64

    dm = KLUEDataModule(
        model_name=model_name,
        file_train=file_train,
        file_val=file_val,
        file_test=file_test,
        file_pred=file_pred,
        data_dir=data_dir,
        batch_size=batch_size,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "train.csv").exists()
    assert Path(data_dir, "val.csv").exists()
    assert Path(data_dir, "test.csv").exists()
    assert Path(data_dir, "pred.csv").exists()

    dm.setup()

    dm.setup("fit")
    assert dm.train_dataloader() and dm.val_dataloader()
    assert isinstance(dm.train_dataloader(), torch.utils.data.DataLoader)

    dm.setup("test")
    assert dm.test_dataloader()

    dm.setup("predict")
    assert dm.predict_dataloader()

    batch = next(iter(dm.train_dataloader()))
    inputs = {key: val for key, val in batch.items() if key != "labels"}
    targets = batch["labels"]

    assert len(inputs["input_ids"]) == batch_size
    assert len(targets) == batch_size
    assert targets.dtype == torch.int64
