import copy
import logging
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger('mylogger')


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        return F.cross_entropy(input, target, weight=weight, reduction=self.reduction)


class CDANLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, use_entropy=True, coeff=1):
        super(CDANLoss, self).__init__()
        self.use_entropy = use_entropy
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.coeff = coeff
        self.entropy_loss = EntropyLoss(coeff=1., reduction='none')

    def forward(self, ad_out, softmax_output=None, coeff=1.0, dc_target=None):
        batch_size = ad_out.size(0) // 2
        dc_target = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), 0).float().to(ad_out.device)
        loss = self.criterion(ad_out.view(-1), dc_target.view(-1))

        if self.use_entropy:
            entropy = self.entropy_loss(softmax_output)
            entropy.register_hook(grl_hook(coeff))
            entropy = 1 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[batch_size:] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[:batch_size] = 0
            target_weight = entropy * target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            return self.coeff*torch.sum(weight * loss) / torch.sum(weight).detach().item()
        else:
            return self.coeff*torch.mean(loss.squeeze())


class EntropyLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, coeff=1., reduction='mean'):
        super().__init__()
        self.coeff = coeff
        self.reduction = reduction

    def forward(self, input):

        epsilon = 1e-5
        entropy = -input * torch.log(input + epsilon)
        entropy = torch.sum(entropy, dim=1)
        if self.reduction == 'none':
            return entropy
        return self.coeff * entropy.mean()



