### losses.py
# Define loss functions.
# Author: Gina Wu @ 01/22
###

import copy
import logging
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import grl_hook, MemoryModule
from .contrastive import MSCLoss

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


class AFNLoss(nn.Module):

    def __init__(self,  coeff=1., offset=0.1):
        """ Refer: https://github.dev/jihanyang/AFN/tree/master/vanilla/Office31/SAFN/code
        """

        super().__init__()
        self.coeff = coeff
        self.offset = offset

    def forward(self, input):

        radius = input.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.offset
        l = ((input.norm(p=2, dim=1) - radius) ** 2).mean()
        return self.coeff * l

class MCCLoss(nn.Module):

    def __init__(self, temperature=2.5, coeff=1):

        super().__init__()
        self.temperature = temperature
        self.coeff = coeff
        self.entropy_loss = EntropyLoss(reduction='none')

    def forward(self, input):
        """ Refer: https://github.com/thuml/Versatile-Domain-Adaptation/blob/master/pytorch/train_image_office.py
        """

        train_bs, class_num = input.shape

        outputs_target_temp = input / self.temperature
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = self.entropy_loss(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num

        return self.coeff*mcc_loss

class MDDLoss(nn.Module):

    def __init__(self, coeff=1):

        super().__init__()
        self.coeff = coeff
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, output_src, output_tgt, output_src_adv, output_tgt_adv):
        """ Refer: https://github.com/thuml/MDD/blob/master/model/MDD.py
        """

        target_adv_src = output_src.max(1)[1].detach()
        target_adv_tgt = output_tgt.max(1)[1].detach()

        classifier_loss_adv_src = self.class_loss(output_src_adv, target_adv_src)

        logloss_tgt = torch.log(1 - F.softmax(output_tgt_adv, dim = 1) + 1e-6)
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        mdd_loss = self.coeff * classifier_loss_adv_src + classifier_loss_adv_tgt

        return mdd_loss

class HDALoss(nn.Module):

    def __init__(self, coeff=1):

        super().__init__()
        self.coeff = coeff

    def forward(self, focals_all):
        """ Refer: https://github.com/microsoft/UDA/blob/main/trainer/da/toalign.py
        """
        hda_loss = self.coeff * focals_all.abs().mean()

        return hda_loss

class MemSACLoss(nn.Module):

    def __init__(self, dim=256, queue_size=48000, momentum=0, coeff=0.1, n_neighb=5, distance="cosine", temperature=0.07, warm_up=4000):

        super().__init__()
        self.memory = MemoryModule(dim, queue_size, momentum)
        self.memory = self.memory.cuda()
        self.msc_loss = MSCLoss(n_neighb, distance, temperature)
        self.coeff = coeff
        self.warm_up = warm_up

    def forward(self, features, source_labels, it):
        """ Refer: https://github.com/ViLab-UCSD/MemSAC_ECCV2022
        """

        ## first, enqueue and dequeue the features.
        source_features = features[:source_labels.shape[0]]
        target_features = features[source_labels.shape[0]:]
        self.memory(source_features, source_labels)

        ## compute the consistency loss using the memory bank
        msac_loss = self.msc_loss(self.memory.queue, self.memory.queue_labels, target_features)

        coeff = (it > self.warm_up) * self.coeff

        return coeff*msac_loss


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



