### metrics.py
# Evaluation metrics.
###


import numpy as np


def percls_accuracy(all_pred, all_label, num_class=0):
    """Computes per class accuracy"""
    num_class = len(set(all_label)) if num_class == 0 else num_class
    all_pred = np.asarray(all_pred)
    all_label = np.asarray(all_label)

    cls_acc = np.zeros([num_class])
    for i in range(num_class):
        idx = (all_label == i)
        if idx.sum() > 0:
            cls_acc[i] = (all_pred[idx] == all_label[idx]).mean() * 100.0

    return cls_acc


def bin_accuracy(output, target):
    """Computes the binary classification accuracy"""
    pred = (output > 0.5).long()
    acc = (pred == target).float().mean() * 100.0
    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

