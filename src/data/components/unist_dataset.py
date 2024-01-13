import random
from ast import literal_eval
from collections import defaultdict

import datasets
from torch.utils.data import Dataset
from tqdm import tqdm


class KLUEDataset(Dataset):
    def __init__(self, dataset, is_pred=False):
        self.data = []
        raw_labelset = [
            "no_relation",
            "org:top_members/employees",
            "org:members",
            "org:product",
            "per:title",
            "org:alternate_names",
            "per:employee_of",
            "org:place_of_headquarters",
            "per:product",
            "org:number_of_employees/members",
            "per:children",
            "per:place_of_residence",
            "per:alternate_names",
            "per:other_family",
            "per:colleagues",
            "per:origin",
            "per:siblings",
            "per:spouse",
            "org:founded",
            "org:political/religious_affiliation",
            "org:member_of",
            "per:parents",
            "org:dissolved",
            "per:schools_attended",
            "per:date_of_death",
            "per:date_of_birth",
            "per:place_of_birth",
            "per:place_of_death",
            "org:founded_by",
            "per:religion",
        ]

        assert isinstance(dataset, datasets.Dataset)

        self.dataset = dataset.map(self.get_entity_dict, batched=True)
        self.dataset = self.dataset.map(self.add_task_description)
        self.is_pred = is_pred
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]

    def preprocess_label(self, label):
        rep_rule = (("_", " "), ("per:", "person "), ("org:", "organization "))
        for r in rep_rule:
            label = label.replace(*r)
        return label

    def get_entity_dict(self, examples):
        return {
            "subject_entity": [literal_eval(example) for example in examples["subject_entity"]],
            "object_entity": [literal_eval(example) for example in examples["object_entity"]],
        }

    def add_task_description(self, examples):
        ss, se, st, sw = (
            examples["subject_entity"]["start_idx"],
            examples["subject_entity"]["end_idx"] + 1,
            examples["subject_entity"]["type"].lower(),
            examples["subject_entity"]["word"],
        )
        os, oe, ot, ow = (
            examples["object_entity"]["start_idx"],
            examples["object_entity"]["end_idx"] + 1,
            examples["object_entity"]["type"].lower(),
            examples["object_entity"]["word"],
        )

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

        desc = f"{st} {sw}와(과) {ot} {ow} 사이의 관계를 설명하시오."

        return {"sentence": sent, "description": desc}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = defaultdict()
        item["sentence"] = self.dataset[idx]["sentence"]
        item["description"] = self.dataset[idx]["description"]
        if not self.is_pred:
            true = self.preprocess_label(self.dataset[idx]["label"])
            item["label"] = true
            false = true
            while false == true:
                false = random.choice(self.labelset)
            item["false"] = false

        return item
