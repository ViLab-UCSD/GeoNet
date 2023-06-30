### __init__.py
# Get loss functions with designated parameters.
# Author: Tarun Kalluri @ 07/22
###

import copy
import torch.nn as nn
import logging

from .loss import AFNLoss, CDANLoss, HDALoss, MCCLoss, EntropyLoss, CrossEntropyLoss, MemSACLoss, MDDLoss

logger = logging.getLogger('mylogger')

def get_loss(loss_dict, verbose=False):

    name = loss_dict['name']
    criterion = _get_loss_instance(name)
    param_dict = copy.deepcopy(loss_dict)
    param_dict.pop('name')

    if 'bce' in name or 'cross_entropy' in name:
        _ = param_dict.setdefault('reduction', 'none')

    criterion = criterion(**param_dict)

    if verbose:
        logger.info('Using {} loss function'.format(name))

    return criterion

def _get_loss_instance(name):
    try:
        return {
            'cross_entropy': CrossEntropyLoss,
            'bce_with_logits': nn.BCEWithLogitsLoss,
            'bce': nn.BCELoss,
            'cdan': CDANLoss,
            'dann': CDANLoss,
            'EntropyLoss': EntropyLoss,
            'AFNLoss': AFNLoss,
            'MCCLoss': MCCLoss,
            'MemSACLoss': MemSACLoss,
            'MDDLoss' : MDDLoss,
            'HDALoss' : HDALoss
        }[name]
    except:
        raise BaseException('Loss function {} not available'.format(name))

