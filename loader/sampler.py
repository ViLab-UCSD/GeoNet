import numpy as np
from collections import Counter
import torch
from torch.utils.data.sampler import Sampler
import logging
import copy


logger = logging.getLogger('mylogger')


def get_sampler(dataset, smplr_dict):

    name = smplr_dict['name']
    logging.info('Using {} sampler'.format(name))

    if name == 'random':
        return None

    param_dict = copy.deepcopy(smplr_dict)
    param_dict.pop('name')
    sampler = _get_sampler_instance(name)
    sampler = sampler(dataset, **param_dict)

    return sampler


def _get_sampler_instance(name):
    try:
        return {
            'class_balanced': BalancedSampler
        }[name]
    except:
        raise BaseException('{} sampler not available'.format(name))


class BalancedSampler(Sampler):
    def __init__(self, dataset, indices=None, tgt_transform=None):

        self.indices = list(range(len(dataset))) if indices is None else indices

        class_ids = np.asarray(dataset.target)[self.indices]
        if tgt_transform is not None:
            class_ids = list(map(tgt_transform, class_ids))

        self.n_samples = len(class_ids)

        # compute class frequencies and set them as sampling weights
        counts = Counter(class_ids)
        get_freq = lambda x: 1.0 / counts[x]
        self.weights = torch.DoubleTensor(list(map(get_freq, class_ids)))

    def __iter__(self):
        sampled_idx = torch.multinomial(self.weights, self.n_samples, replacement=True)
        return (self.indices[i] for i in sampled_idx)

    def __len__(self):
        return self.n_samples


