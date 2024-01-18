import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        if self.use_activation:
            nn.init.normal_(self.linear.weight)
        else:
            nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.gelu(x)
        return self.linear(x)


class RBERT(PreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config)
        self.plm = AutoModel.from_pretrained(model_name, config=config)

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.dropout_rate)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            config.dropout_rate,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """Average the entity hidden state vectors (H_i ~ H_j) :param hidden_output: [batch_size,
        j-i+1, dim] :param e_mask: [batch_size, max_seq_len] e.g. e_mask[0] == [0, 0, 0, 1, 1, 1,
        0, 0, ...

        0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, subject_mask, object_mask):
        outputs = self.plm(
            input_ids=input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs.last_hidden_state
        cls_output = outputs.get("pooler_output", outputs.last_hidden_state[:, 0])  # [CLS]

        # Average
        subject_h = self.entity_average(sequence_output, subject_mask)
        object_h = self.entity_average(sequence_output, object_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer of e1 and e2)
        pooled_output = self.cls_fc_layer(cls_output)
        subject_h = self.entity_fc_layer(subject_h)
        object_h = self.entity_fc_layer(object_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, object_h, subject_h], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits
