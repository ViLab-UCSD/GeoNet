### advnet.py
# Ref: https://github.com/thuml/CDAN/blob/master/pytorch/network.py
###

import numpy as np
import torch.nn as nn
from .utils import grl_hook


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)

    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer2.bias.data.fill_(0.0)

    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer3.bias.data.fill_(0.0)

    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    # self.sigmoid = nn.Sigmoid()

  def forward(self, x, coeff):
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    # y = self.sigmoid(y)
    return y


def advnet(**kwargs):
    model = AdversarialNetwork(**kwargs)
    return model


