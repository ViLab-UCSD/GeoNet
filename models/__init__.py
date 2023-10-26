import copy
import torch.nn as nn
import logging

from .resnet import resnet10, resnet101, resnet18, resnet50
from .mlpcls import mlpcls
from .advnet import advnet
from .randomlyr import randomlayer
from .utils import grl_hook


logger = logging.getLogger('mylogger')


def get_model(model_dict, verbose=False):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if 'resnet' in name:
        model = model(**param_dict)
        model.fc = nn.Identity()
    else:
        model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'mlpcls': mlpcls,
            'advnet': advnet,
            'randomlyr': randomlayer,
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


