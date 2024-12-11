# adopt from https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py

import torch
import torch.nn.functional as F
from torch import nn
from ..builder import LOSSES

@LOSSES.register_module()
class JSDivergenceLoss(nn.Module):  
    def __init__(self):  
        super(JSDivergenceLoss, self).__init__()  
      
    def kl_divergence(self, p, q):  
        # Ensure q is a valid probability distribution  
        q = F.log_softmax(q, dim=1)  
        # Compute KL divergence  
        kl_div = F.kl_div(q, p, reduction='batchmean')  
        return kl_div  
      
    def forward(self, p, q):  
        # Ensure p and q are valid probability distributions  
        p = F.softmax(p, dim=1)  
        q = F.softmax(q, dim=1)  
          
        # Compute the mixture distribution m  
        m = (p + q) / 2.0  
          
        # Compute KL(p || m) and KL(q || m)  
        kl_p_m = self.kl_divergence(m, p)  
        kl_q_m = self.kl_divergence(m, q)  
          
        # JS divergence is the average of these two KL divergences  
        js_div = (kl_p_m + kl_q_m) / 2.0  
          
        return js_div  