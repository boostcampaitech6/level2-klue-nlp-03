import torch.nn as nn
import torch
import torch.nn.functional as F

class SoftLoss(nn.Module):
    def __init__(self, eps=0.3, pad_mask=-1):
        super(SoftLoss, self).__init__()
        self.eps = eps
        self.pad_mask = pad_mask

    def forward(self, inputs, targets):
        n_class = inputs.size(1)
        one_hot = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
        log_prb = F.log_softmax(inputs, dim=1)
        non_pad_mask = targets.ne(self.pad_mask)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
        return loss/len(inputs)