### mlpcls.py
# Module of FewShot classifier.
# Refer https://github.com/virajprabhu/SENTRY/blob/main/adapt/models/task_net.py
# Author: Tarun Kalluri @ 07/22
###

import torch
from torch import nn
import torch.nn.functional as F


class fewShotCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, temperature=0.05):
        super(fewShotCls, self).__init__()
        self.n_class = n_class
        self.temperature = temperature

        assert len(feat_size) == 1 , "For few shot classifier, MLP is not allowed."

        self.out = nn.Linear(feat_size[-1], n_class, bias=False)
        torch.nn.init.xavier_normal_(self.out.weight)


    def forward(self, x, feat=False):

        assert feat == False

        x = F.normalize(x, p=2, dim=1)

        y = self.out(x) / self.temperature

        if feat:
            return y, x
        return y


def fscls(**kwargs):
    model = fewShotCls(**kwargs)
    return model


if __name__ == '__main__':
    import pdb
    model = fscls(**{'feat_size': [2048], 'n_class': 2})
    x = torch.randn([4, 2048])
    y = model(x)
    pdb.set_trace()


