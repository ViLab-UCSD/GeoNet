import logging
import copy
from torch.optim import SGD, Adam, lr_scheduler

from .scheduler import MultiStep_scheduler, step_scheduler, inv_scheduler

logger = logging.getLogger('mylogger')

def get_optimizer(opt_dict):
    """Function to get the optimizer instance.
    """
    name = opt_dict['name']
    optimizer = _get_opt_instance(name)
    param_dict = copy.deepcopy(opt_dict)
    param_dict.pop('name')
    logger.info('Using {} optimizer'.format(name))

    return optimizer, param_dict

def _get_opt_instance(name):
    try:
        return {
            'sgd': SGD,
            'adam': Adam,
        }[name]
    except:
        raise ('Optimizer {} not available'.format(name))

def get_scheduler(optim, schdlr_dict):

    name = schdlr_dict['name']
    scheduler = _get_schdlr_instance(name)
    param_dict = copy.deepcopy(schdlr_dict)
    param_dict.pop('name')

    return scheduler(optimizer=optim, **param_dict)

def _get_schdlr_instance(name):
    try:
        return {
            'stepLR': step_scheduler,
            'inv': inv_scheduler,
            'multiStepLr': MultiStep_scheduler
        }[name]
    except:
        raise ('LR scheduler {} not available'.format(name))


