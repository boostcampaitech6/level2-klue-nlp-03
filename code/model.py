import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig


# model for Trainer
class Model(nn.Module):
    def __init__(self, model_name, num_labels):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.plm = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.loss = nn.CrossEntropyLoss()
        self.save_pretrained = self.plm.save_pretrained

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if labels is not None:
            loss = self.loss(outputs.logits, labels)
            return {'loss': loss, 'logits': outputs.logits }
        return {'logits': outputs.logits }


        
