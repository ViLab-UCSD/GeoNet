### mlpcls.py
# Module of MLP classifier.
# Author: Gina Wu @ 01/22
###

import torch
from torch import nn


class HDACls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, nonlinear='relu', bn=True, dp=False, dropout_p=0.5,
                    toalign=True, hda=True):
        """ 
            https://github.com/microsoft/UDA/blob/main/models/base_model.py
        """
        super(HDACls, self).__init__()
        self.n_class = n_class
        self.norm = norm
        self.dp = dp
        self.dropout_p = dropout_p

        self.toalign = toalign
        self.hda = hda

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

        self.fc = nn.Linear(feat_size[-1], n_class)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

        # HDA head
        self._fdim = feat_size[-1]
        if self.hda:
            self.fc.apply(init_weights_fc)
            self.fc0 = nn.Linear(self._fdim, n_class)
            self.fc0.apply(init_weights_fc0)
            self.fc1 = nn.Linear(self._fdim, n_class)
            self.fc1.apply(init_weights_fc1)
            self.fc2 = nn.Linear(self._fdim, n_class)
            self.fc2.apply(init_weights_fc2)

    def forward(self, x, toalign=False, labels=None, feat=False):

        x = self.mlp(x)

        toalign = toalign * self.toalign ## both global and local settings should be true

        if toalign:
            w_pos = self._get_toalign_weight(x, labels=labels)
            f_pos = x * w_pos
            feature = f_pos
            y_pos = self.fc(f_pos)
            if self.hda:
                z_pos0 = self.fc0(f_pos)
                z_pos1 = self.fc1(f_pos)
                z_pos2 = self.fc2(f_pos)
                z_pos = z_pos0 + z_pos1 + z_pos2
                logits = y_pos - z_pos
                residual = z_pos
            else:
                logits = y_pos
                residual = None
        else:
            y = self.fc(x)
            feature = x
            if self.hda:
                z0 = self.fc0(x)
                z1 = self.fc1(x)
                z2 = self.fc2(x)
                z = z0 + z1 + z2
                logits = y - z
                residual = z
            else:
                logits = y
                residual = None

        if not self.training:
            return logits
        elif self.hda:
            return feature, logits, residual
        else:
            return feature, logits, residual

        

    def _get_toalign_weight(self, f, labels):
        """https://github.com/microsoft/UDA/blob/main/models/base_model.py
        """
        assert labels is not None, f'labels should be asigned'
        w = self.fc.weight[labels].detach()  # [B, C]
        if self.hda:
            w0 = self.fc0.weight[labels].detach()
            w1 = self.fc1.weight[labels].detach()
            w2 = self.fc2.weight[labels].detach()
            w = w - (w0 + w1 + w2)
        eng_org = (f**2).sum(dim=1, keepdim=True)  # [B, 1]
        eng_aft = ((f*w)**2).sum(dim=1, keepdim=True)  # [B, 1]
        scalar = ((eng_org + 1e-6) / (eng_aft + 1e-6)).sqrt()
        w_pos = w * scalar
        # print(scalar)

        return w_pos


def hdacls(**kwargs):
    model = HDACls(**kwargs)
    return model


if __name__ == '__main__':
    import pdb
    model = hdacls(**{'feat_size': [2048, 512], 'n_class': 2})
    x = torch.randn([4, 2048])
    y = model(x)
    pdb.set_trace()


# initialization used only for HDA
def init_weights_fc(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=100)
        nn.init.zeros_(m.bias)


def init_weights_fc0(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def init_weights_fc1(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1)
        nn.init.zeros_(m.bias)


def init_weights_fc2(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=2)
        nn.init.zeros_(m.bias)
