import math
import torch
from torch.autograd import Function
import torch.nn as nn

from ..builder import MODELS

class GradReverse(Function):  
    @staticmethod  
    def forward(ctx, x, lambda_):  
        ctx.lambda_ = lambda_  
        return x.view_as(x)  
  
    @staticmethod  
    def backward(ctx, grad_output):  
        return grad_output * -ctx.lambda_, None  

@MODELS.register_module()
class GradientReversalLayer(nn.Module):  
    def __init__(self, lambda_=1.0):  
        super(GradientReversalLayer, self).__init__()  
        self.lambda_ = lambda_  
  
    def forward(self, x):  
        return GradReverse.apply(x, self.lambda_)
    
    