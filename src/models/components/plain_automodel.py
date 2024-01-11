import torch.nn as nn
from transformers import AutoConfig, AutoModel


# model for Trainer
class Encoder(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.plm = AutoModel.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_outputs = outputs.get("pooler_output", outputs.last_hidden_state[:, 0])
        return pooled_outputs
