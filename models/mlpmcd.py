### mlpcls.py
# Module of MLP classifier.
# Author: Gina Wu @ 01/22
###

import torch
from torch import nn

class MCDCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, nonlinear='relu', bn=True, dp=False, _sqrt_norm=False, dropout_p=0.5):
        """ 
            https://github.com/thuml/MDD
        """
        super(MCDCls, self).__init__()
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
            if bn:
                layers.append(nn.BatchNorm1d(feat_size[i+1]))
            if nonlinear != 'none':
                layers.append(nn.ReLU(inplace=True))
            if dp:
                layers.append(nn.Dropout(self.dropout_p))

        self.mlp = nn.Sequential(*layers)

        self.out_1 = nn.Linear(feat_size[-1], n_class)

        self.out_2 = nn.Linear(feat_size[-1], n_class)

        self.apply(weights_init)

    def forward(self, x, feat=False):

        x = self.mlp(x)

        output_1 = self.out_1(x)
        output_2 = self.out_2(x)
        
        if self.training:
            return output_1, output_2
        else:
            return output_1 + output_2

def mcdcls(**kwargs):
    model = MCDCls(**kwargs)
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
    model = mcdcls(**{'feat_size': [2048, 512, 128], 'n_class': 2})
    x = torch.randn([4, 2048])
    y = model(x)
    pdb.set_trace()




