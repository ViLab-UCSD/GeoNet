### utils.py
# Utility functions.
###

import torch
import numpy as np

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


# class GradientReverseLayer(torch.autograd.Function):
#     def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
#         self.iter_num = iter_num
#         self.alpha = alpha
#         self.low_value = low_value
#         self.high_value = high_value
#         self.max_iter = max_iter

#     @staticmethod
#     def forward(ctx, input):
#         ctx.iter_num += 1
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(self, grad_output):
#         self.coeff = calc_coeff(self.iter_num, self.high_value, self.low_value, self.alpha, self.max_iter)
#         return -self.coeff * grad_output

class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 1000
    @staticmethod
    def forward(ctx, input):
        GradientReverseLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 1
        low = 0.0
        high = 0.1
        iter_num, max_iter = GradientReverseLayer.iter_num, GradientReverseLayer.max_iter 
        coeff = calc_coeff(iter_num, high, low, alpha, max_iter)
        return -coeff * gradOutput
