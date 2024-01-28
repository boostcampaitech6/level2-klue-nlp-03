from typing import Any, Dict, Optional

import pickle
from ast import literal_eval

from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
from models.components.minority_classes import REF_SENT, MINOR_LABELS
import torch

"""❤@#^"""
SUB_PUNCT_IN = "❤"
SUB_PUNCT_OUT = "#"
OBJ_PUNCT_IN = "@"
OBJ_PUNCT_OUT = "^"
SUB_START_PUNCT = SUB_PUNCT_OUT + SUB_PUNCT_IN
OBJ_START_PUNCT = OBJ_PUNCT_OUT + OBJ_PUNCT_IN

ko_entity_type = {
    "PER": "사람",
    "ORG": "단체",
    "DAT": "날짜",
    "LOC": "장소",
    "POH": "기타",
    "NOH": "수량",
}

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
    
    def semantic_typing(self, examples):
        return {
            "desc": [
                f"{SUB_START_PUNCT} {ko_entity_type[set]} {SUB_PUNCT_IN} {s[sesi:seei + 1]} {SUB_PUNCT_OUT} 과 + {OBJ_START_PUNCT} {ko_entity_type[oet]} {OBJ_PUNCT_IN} {s[oesi:oeei + 1]} {OBJ_PUNCT_OUT} 의 관계는 무엇인가?"
                if sesi < oesi else
                f"{OBJ_START_PUNCT} {ko_entity_type[oet]} {OBJ_PUNCT_IN} {s[oesi:oeei + 1]} {OBJ_PUNCT_OUT} 과 + {SUB_START_PUNCT} {ko_entity_type[set]} {SUB_PUNCT_IN} {s[sesi:seei + 1]} {SUB_PUNCT_OUT} 의 관계는 무엇인가?"
                for s, set, oet, sesi, seei, oesi, oeei in zip(
                    examples["sentence"],
                    examples["subject_entity_type"], examples["object_entity_type"],
                    examples["subject_entity_start_idx"], examples["subject_entity_end_idx"],
                    examples["object_entity_start_idx"], examples["object_entity_end_idx"]
                )     
            ],
        }
    
    def get_ref_inputids(self, ref_sent: list):
            input_ids_ls = [self.tokenizer.encode(sent) for sent in ref_sent]
            max_len = max([len(ls) for ls in input_ids_ls])
            input_ids = [ls + [0]*(max_len-len(ls)) for ls in input_ids_ls]
            input_mask = [ [1.0]*len(ls) + [0]*(max_len-len(ls)) for ls in input_ids_ls]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float)
            return input_ids, input_mask
        
    def tokenize_ref_sent(self):
        minor_label_ids = [self.label2id[label] for label in MINOR_LABELS]
        ref_labels_id = minor_label_ids
        ref_labels_id = sorted(ref_labels_id)
        ref_sent = [REF_SENT[i] for i in ref_labels_id]
        ref_ids, ref_mask = self.get_ref_inputids(ref_sent)
        return ref_ids, ref_mask

    def customize_token_type_ids(self, examples):
        embedding_token_type_ids_list = []
        sub_idxs = []
        obj_idxs = []
        for idx in tqdm(range(len(examples["input_ids"]))):
            is_sub_start = False
            is_obj_start = False
            SUB_TOKEN = 1
            sub_idx = 600
            obj_idx = 700
            OBJ_TOKEN = 2
            ELSE_TOKEN = 0
            embedding_token_type_ids = []
            for i in range(len(examples["input_ids"][idx])):
                if is_sub_start == True:
                    sub_idx = min(sub_idx, i - 1)
                    embedding_token_type_ids[-1] = SUB_TOKEN
                    embedding_token_type_ids.append(SUB_TOKEN)
                elif is_obj_start == True:
                    obj_idx = min(obj_idx, i - 1)
                    embedding_token_type_ids[-1] = OBJ_TOKEN
                    embedding_token_type_ids.append(OBJ_TOKEN)
                else:
                    embedding_token_type_ids.append(ELSE_TOKEN)
                
                if self.tokenizer.decode(examples["input_ids"][idx][i]) == SUB_START_PUNCT:
                    is_sub_start = not is_sub_start
                elif self.tokenizer.decode(examples["input_ids"][idx][i]) == OBJ_START_PUNCT:
                    is_obj_start = not is_obj_start
                elif self.tokenizer.decode(examples["input_ids"][idx][i]) == SUB_PUNCT_OUT:
                    is_sub_start = not is_sub_start
                elif self.tokenizer.decode(examples["input_ids"][idx][i]) == OBJ_PUNCT_OUT:
                    is_obj_start = not is_obj_start

            if is_sub_start != False or is_obj_start != False:
                print("error")
                raise Exception("error")

            embedding_token_type_ids_list.append(embedding_token_type_ids)
            sub_idxs.append(sub_idx)
            obj_idxs.append(obj_idx)
            
        return {
            "embedding_token_type_ids_list": embedding_token_type_ids_list,
            "sub_idxs": sub_idxs,
            "obj_idxs": obj_idxs,
        }
    
    def guidance(self, examples):
        return {
            "attn_guide" : [
                1 if example in MINOR_LABELS else 0
                for example in examples["label"]
            ]
        }

    def tokenize_function(self, examples):
        # new_tokens = [SUB_PUNCT_IN, SUB_PUNCT_OUT, OBJ_PUNCT_IN, OBJ_PUNCT_OUT]
        # self.tokenizer.add_tokens(new_tokens)
        return self.tokenizer(examples["desc"] ,examples["sentence"], truncation=True)

    def preprocessing(self, dataset, stage):
        dataset = dataset.map(self.get_word_word_type_and_idxs_from_entity, batched=True)
        dataset = dataset.map(self.semantic_typing, batched=True)
        dataset = dataset.map(self.tokenize_function, batched=True)
        dataset = dataset.map(self.customize_token_type_ids, batched=True)
        dataset = dataset.map(self.guidance, batched=True)
        dataset = dataset.select_columns(
            ["input_ids", "attention_mask", "token_type_ids", "sub_idxs", "obj_idxs", "label"]
        )
        # dataset = dataset.select_columns(
        #     ["input_ids", "token_type_ids", "attention_mask", "label"]
        # )
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
