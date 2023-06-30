### utils.py
# Functions for logging, loading weights, assign learning rates.
###

import os, sys
import logging
import datetime
import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def parameters_in_module(module):
    return sum([p.numel() for p in module.parameters()])

def get_parameter_count(params):
    total_count = 0
    for pg in params:
        for module in pg["params"]:
            total_count += sum([m.numel() for m in module])
    return total_count

def assign_learning_rate(optimizer, lr=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def add_weight_decay(params, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def get_logger(logdir, test=False):
    """Function to build the logger.
    """
    logger = logging.getLogger('mylogger')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    if test: ts = "test"
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_hdlr.setFormatter(formatter)
    logger.addHandler(file_hdlr)
    logger.propagate = False
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    stdout_hdlr.setFormatter(formatter)
    logger.addHandler(stdout_hdlr)
    logger.setLevel(logging.INFO)
    return logger


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
    module state_dict inplace, i.e. removing "module" in the string.
    """
    return {k.partition("module.")[-1]:v for k,v in state_dict.items()}