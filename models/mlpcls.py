### mlpcls.py
# Module of MLP classifier.
# Author: Gina Wu @ 01/22
###

from models.utils import GradientReverseLayer
import torch
from torch import nn
import torch.nn.functional as F
import math


class MLPCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, nonlinear='relu', bn=True, dp=False, _sqrt_norm=False, dropout_p=0.5):
        """ 
            _sqrt_norm or use in AFN (dp must be True). Refer: https://github.dev/jihanyang/AFN/tree/master/vanilla/Office31/SAFN/code
        """
        super(MLPCls, self).__init__()
        self.n_class = n_class
        self.norm = norm
        self.dp = dp
        self.dropout_p = dropout_p
        self._sqrt_norm = _sqrt_norm
        if self._sqrt_norm: 
            assert dp , "DropOut must be enabled for using sqrt norm."

        layers = []
        for i in range(len(feat_size)-1):
            layers.append(nn.Linear(feat_size[i], feat_size[i+1]))
            layers[-1].weight.data.normal_(0, 0.005)
            layers[-1].bias.data.fill_(0.1)
            if bn:
                layers.append(nn.BatchNorm1d(feat_size[i+1]))
            if nonlinear != 'none':
                layers.append(nn.ReLU(inplace=True))
            if dp:
                layers.append(nn.Dropout(self.dropout_p))

        self.mlp = nn.Sequential(*layers)

        if self.norm > 0:
            self.out = nn.Linear(feat_size[-1], n_class, bias=False)
        else:
            self.out = nn.Linear(feat_size[-1], n_class)
            self.out.weight.data.normal_(0, 0.01)
            self.out.bias.data.fill_(0.0)

        # self.apply(weights_init)

    def forward(self, x, feat=False):
        
        x = self.mlp(x)

        if self.norm > 0:
            normalized_feats = F.normalize(x, p=2, dim=1) * self.norm
            weight = F.normalize(self.out.weight, p=2, dim=1) * self.norm
            y = torch.mm(normalized_feats, weight.t())
        elif self._sqrt_norm:
            if self.training:
                x.mul_(math.sqrt(1 - self.dropout_p))
            y = self.out(x)
        else:
            y = self.out(x)

        if feat:
            return y, x
        return y


def mlpcls(**kwargs):
    model = MLPCls(**kwargs)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)



if __name__ == '__main__':
    import pdb
    model = mlpcls(**{'feat_size': [2048, 512, 128], 'n_class': 2})
    x = torch.randn([4, 2048])
    y = model(x)
    pdb.set_trace()




