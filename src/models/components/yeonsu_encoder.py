import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


# model for Trainer
class YeonsuEncoder(nn.Module):
    def __init__(self, model_name, num_labels, do_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.plm = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout1 = nn.Dropout(do_prob)
        self.lstm = nn.LSTM(self.plm.config.hidden_size, 256)
        self.dropout2 = nn.Dropout(do_prob)
        self.fc = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout1(cls_output)
        lstm_output, _ = self.lstm(cls_output)
        lstm_output = self.dropout2(lstm_output)
        logits = self.fc(lstm_output)

        return logits
