from ast import literal_eval

import datasets
from torch.utils.data import Dataset

from .labelsets import label2id, type_translated


class SemanticTypingDataset(Dataset):
    def __init__(self, dataset, is_pred=False):
        assert isinstance(dataset, datasets.Dataset)

        self.dataset = dataset.map(self.get_entity_dict, batched=True)
        self.dataset = self.dataset.map(self.add_task_description)
        self.is_pred = is_pred
        if not self.is_pred:
            self.dataset = self.dataset.map(self.encode_label_to_id, batched=True)

    def get_entity_dict(self, examples):
        return {
            "subject_entity": [literal_eval(example) for example in examples["subject_entity"]],
            "object_entity": [literal_eval(example) for example in examples["object_entity"]],
        }

    def add_task_description(self, examples):
        ss, se, st, sw = (
            examples["subject_entity"]["start_idx"],
            examples["subject_entity"]["end_idx"] + 1,
            examples["subject_entity"]["type"],
            examples["subject_entity"]["word"],
        )
        os, oe, ot, ow = (
            examples["object_entity"]["start_idx"],
            examples["object_entity"]["end_idx"] + 1,
            examples["object_entity"]["type"],
            examples["object_entity"]["word"],
        )

        st = type_translated[st]
        ot = type_translated[ot]

        if ss < os:
            sent = (
                examples["sentence"][:ss]
                + "<SUBJ> "
                + st
                + " "
                + examples["sentence"][ss:se]
                + " </SUBJ>"
                + examples["sentence"][se:os]
                + "<OBJ> "
                + ot
                + " "
                + examples["sentence"][os:oe]
                + " </OBJ>"
                + examples["sentence"][oe:]
            )
        else:
            sent = (
                examples["sentence"][:os]
                + "<OBJ> "
                + ot
                + " "
                + examples["sentence"][os:oe]
                + " </OBJ>"
                + examples["sentence"][oe:ss]
                + "<SUBJ> "
                + st
                + " "
                + examples["sentence"][ss:se]
                + " </SUBJ>"
                + examples["sentence"][se:]
            )

        desc = f"<SUBJ> {st} {sw} </SUBJ>와(과) <OBJ> {ot} {ow} </OBJ>사이의 <RELATION>를 설명하시오."

        return {"sentence": sent, "description": desc}

    def encode_label_to_id(self, examples):
        return {"labels": [label2id[example] for example in examples["label"]]}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        item["sentence"] = self.dataset[idx]["sentence"]
        item["description"] = self.dataset[idx]["description"]
        if not self.is_pred:
            item["labels"] = self.dataset[idx]["labels"]

        return item
