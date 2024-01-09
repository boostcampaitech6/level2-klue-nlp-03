from typing import Any, Dict, List, Tuple

import os
import pickle

import hydra
import pandas as pd
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def inference(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("predict")
    predict_dataloader = datamodule.predict_dataloader()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting Inference!")
    logits = trainer.predict(model=model, dataloaders=predict_dataloader, ckpt_path=cfg.ckpt_path)

    id2label = pd.read_pickle(cfg.paths.data_dir + "id2label.pkl")

    logits = torch.concat(logits).detach().cpu()
    probs = torch.nn.functional.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    pred_label = [id2label[int(idx)] for idx in preds]
    output = pd.DataFrame(
        {"id": range(len(pred_label)), "pred_label": pred_label, "probs": probs.tolist()}
    )

    if not os.path.exists(cfg.paths.predict_dir):
        os.makedirs(cfg.paths.predict_dir)

    output.to_csv(cfg.paths.predict_dir + "submission.csv", index=False)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    inference(cfg)


if __name__ == "__main__":
    main()
