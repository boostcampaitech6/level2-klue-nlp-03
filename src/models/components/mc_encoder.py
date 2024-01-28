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

        self.ref_value = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.key_value = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.pool_cat = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)
        self.gate = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, num_labels)

        self._init_weights(self.ref_value)
        self._init_weights(self.key_value)
        self._init_weights(self.pool_cat)
        self._init_weights(self.gate)
        self._init_weights(self.out)

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

    def rbert(self, outputs, sub_idxs, obj_idxs):
        cls_output = outputs.get("pooler_output", outputs.last_hidden_state[:, 0, :])
        sub_incidices = sub_idxs.unsqueeze(1).unsqueeze(-1)
        obj_incidices = obj_idxs.unsqueeze(1).unsqueeze(-1)

        sub_incidices = sub_incidices.expand(outputs.last_hidden_state.size(0), sub_incidices.size(1), outputs.last_hidden_state.size(-1))
        obj_incidices = obj_incidices.expand(outputs.last_hidden_state.size(0), obj_incidices.size(1), outputs.last_hidden_state.size(-1))

        sub_outputs = torch.gather(outputs.last_hidden_state, 1, sub_incidices)
        obj_outputs = torch.gather(outputs.last_hidden_state, 1, obj_incidices)

        sub_outputs = sub_outputs.squeeze(1)
        obj_outputs = obj_outputs.squeeze(1)

        return cls_output, sub_outputs, obj_outputs

    def forward(self, input_ids, attention_mask, token_type_ids, sub_idxs, obj_idxs, attn_guide):
        self.n_batch = input_ids.size(0)

        outputs = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output, sub_outputs, obj_outputs = self.rbert(outputs, sub_idxs, obj_idxs)

        ref_cls_token = self.plm(self.ref_ids, self.ref_mask).last_hidden_state[:, 0, :]
        ref_value = self.ref_value(ref_cls_token)
        ref_key = self.key_value(cls_output)

        concat_outputs = torch.cat([cls_output, sub_outputs, obj_outputs], dim=-1)
        pooled = nn.functional.relu(self.pool_cat(concat_outputs))
        gating = torch.tanh(self.gate(pooled))

        attn_score = torch.matmul(pooled, ref_key.T)/(self.plm.config.hidden_size**(1/2))

        ref_n_class = attn_score.shape[1]
        weighted_ref = ref_value.repeat(self.n_batch, 1, 1)
        weighted_sum_vec = torch.sum(weighted_ref, dim=-1)

        attn_score2 = torch.matmul(pooled, ref_key.T)/(self.plm.config.hidden_size**(1/2))
        attn_score2[attn_guide == 1]
        weighted_sum_vec2 = torch.sum(ref_value.repeat(self.batch_n, 1, 1) * attn_score2.contiguous().view(self.batch_n, ref_n_class, 1), dim=-1)

        cat = torch.cat([pooled + gating*weighted_sum_vec, pooled + gating*weighted_sum_vec2, ref_value], dim=0)
        out = self.out(cat)

        return out[:self.n_batch], torch.softmax(out[self.n_batch:-ref_n_class], dim=-1), out[-ref_n_class:], attn_score, gating
