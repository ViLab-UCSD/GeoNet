### mlpcls.py
# Module of MLP classifier.
# Author: Gina Wu @ 01/22
###

import torch
from torch import nn
import torch.nn.functional as F
import math
from models.utils import GradientReverseLayer


class MDDCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, nonlinear='relu', bn=True, dp=False, _sqrt_norm=False, dropout_p=0.5):
        """ 
            https://github.com/thuml/MDD
        """
        super(MDDCls, self).__init__()
        self.n_class = n_class
        self.norm = norm
        self.dp = dp
        self.dropout_p = dropout_p
        self._sqrt_norm = _sqrt_norm
        if self._sqrt_norm: 
            assert dp , "DropOut must be enabled for using sqrt norm."

        layers = []
        for i in range(len(feat_size)-2):
            layers.append(nn.Linear(feat_size[i], feat_size[i+1]))
            if bn:
                layers.append(nn.BatchNorm1d(feat_size[i+1]))
            if nonlinear != 'none':
                layers.append(nn.ReLU(inplace=True))
            if dp:
                layers.append(nn.Dropout(self.dropout_p))

        self.mlp = nn.Sequential(*layers)

        self.GradRev = GradientReverseLayer()

        classifier_list_1 = [nn.Linear(feat_size[-2], feat_size[-1]), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(feat_size[-1], n_class)]
        self.out_1 = nn.Sequential(*classifier_list_1)

        classifier_list_2 = [nn.Linear(feat_size[-2], feat_size[-1]), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(feat_size[-1], n_class)]
        self.out_2 = nn.Sequential(*classifier_list_2)

        ## initialization: https://github.com/thuml/MDD/blob/51e54847da013ee8f118b79adcce87dbc6e214bf/model/MDD.py#L45
        self.mlp[0].weight.data.normal_(0, 0.005)
        self.mlp[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.out_2[dep * 3].weight.data.normal_(0, 0.01)
            self.out_2[dep * 3].bias.data.fill_(0.0)
            self.out_1[dep * 3].weight.data.normal_(0, 0.01)
            self.out_1[dep * 3].bias.data.fill_(0.0)

    def forward(self, x, feat=False):

        x = self.mlp(x)

        x_adv = self.GradRev.apply(x)
        output_adv = self.out_2(x_adv)
        
        output = self.out_1(x)

        if self.training:
            return output, output_adv
        else:
            return output


def mddcls(**kwargs):
    model = MDDCls(**kwargs)
    return model


if __name__ == '__main__':
    import pdb
    model = mddcls(**{'feat_size': [2048, 512, 128], 'n_class': 2})
    x = torch.randn([4, 2048])
    y = model(x)
    pdb.set_trace()




