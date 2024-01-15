import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


# model for Trainer
class PretrainedEncoder(nn.Module):
    def __init__(self, model_name, num_labels, do_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits
