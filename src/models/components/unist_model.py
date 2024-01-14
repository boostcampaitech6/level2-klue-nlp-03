import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel


class UniSTModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.margin = config.margin
        self.init_weights()

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

        # print("inputs", texts_embeddings, "*"*100)

        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_fn, margin=self.model.config.margin
        )

        loss = loss_fn(texts_embeddings, labels_embeddings, false_embeddings)
        # print("loss", loss, "*"*100)

        return loss, texts_embeddings

    def embed(
        self,
        input_ids,
        attention_mask=None,
    ):
        print("*" * 100, "model", self.model.config)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_outputs = outputs.get("pooler_output", outputs.last_hidden_state[:, 0])
        return pooled_outputs

    def dist_fn(self, texts_embeddings, label_embeddings):
        return 1.0 - F.cosine_similarity(texts_embeddings, label_embeddings)
