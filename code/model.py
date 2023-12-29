import torch.nn as nn
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

class RobertaBiLSTM(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(MODEL_NAME)
        self.model_config.num_labels = 30
        self.model = AutoModel.from_pretrained(MODEL_NAME, config=self.model_config)
        self.hidden_dim = self.model_config.hidden_size
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden, (last_hidden, last_cell) = self.lstm(output)
        output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        logits = self.fc(output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model_config.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

        
