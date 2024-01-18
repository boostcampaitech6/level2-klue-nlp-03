import torch.nn as nn
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from data.remc_datamodule import tokenize_ref_sent


# model for Trainer
class MCEncoder(nn.Module):
    def __init__(self, model_name, num_labels, do_prob=0.1):
        super().__init__()
        self.ref_ids, self.ref_mask = tokenize_ref_sent()
        self.config = AutoConfig.from_pretrained(model_name)
        self.plm = AutoModel.from_pretrained(model_name, config=self.config)
        self.fc1 = nn.Linear(self.plm.config.hidden_size * 3, self.plm.config.hidden_size * 2)
        self.fc2 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.fc3 = nn.Linear(self.plm.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(do_prob)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(do_prob)
        self.bn1 = nn.BatchNorm1d(self.plm.config.hidden_size * 3)
        self.bn2 = nn.BatchNorm1d(self.plm.config.hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(self.plm.config.hidden_size)
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)

    def _init_weights(self, module):
        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids, sub_idxs, obj_idxs):
        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.get("pooler_output", outputs.last_hidden_state[:, 0, :])

        sub_incidices = sub_idxs.unsqueeze(1).unsqueeze(-1)
        obj_incidices = obj_idxs.unsqueeze(1).unsqueeze(-1)

        sub_incidices = sub_incidices.expand(input_ids.size(0), sub_incidices.size(1), outputs.last_hidden_state.size(-1))
        obj_incidices = obj_incidices.expand(input_ids.size(0), obj_incidices.size(1), outputs.last_hidden_state.size(-1))

        sub_outputs = torch.gather(outputs.last_hidden_state, 1, sub_incidices)
        obj_outputs = torch.gather(outputs.last_hidden_state, 1, obj_incidices)

        sub_outputs = sub_outputs.squeeze(1)
        obj_outputs = obj_outputs.squeeze(1)

        concat_outputs = torch.cat([cls_output, sub_outputs, obj_outputs], dim=-1)
        concat_outputs = self.bn1(concat_outputs)
        concat_outputs = self.dropout(concat_outputs)

        fc1_output = self.dropout(self.gelu(self.bn2(self.fc1(concat_outputs))))
        fc2_output = self.dropout(self.gelu(self.bn3(self.fc2(fc1_output))))
        logits = self.fc3(fc2_output)

        return logits
