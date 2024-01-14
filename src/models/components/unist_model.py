import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class UniSTModel(nn.Module):
    def __init__(self, model_name="klue/roberta-base", margin=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.margin = margin
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens_dict = {
            "additional_special_tokens": ["<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        texts_input_ids,
        labels_input_ids,
        false_input_ids,
        texts_attention_mask=None,
        labels_attention_mask=None,
        false_attention_mask=None,
    ):
        texts_embeddings = self.embed(texts_input_ids, texts_attention_mask)
        labels_embeddings = self.embed(labels_input_ids, labels_attention_mask)
        false_embeddings = self.embed(false_input_ids, false_attention_mask)

        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_fn, margin=self.model.config.margin
        )

        loss = loss_fn(texts_embeddings, labels_embeddings, false_embeddings)

        return loss, texts_embeddings

    def embed(
        self,
        input_ids,
        attention_mask=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_outputs = outputs.get("pooler_output", outputs.last_hidden_state[:, 0])
        return pooled_outputs

    def dist_fn(self, texts_embeddings, label_embeddings):
        return 1.0 - F.cosine_similarity(texts_embeddings, label_embeddings)
