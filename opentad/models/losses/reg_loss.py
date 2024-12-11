# adopt from https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py

import torch
from torch import nn
from ..builder import LOSSES

@LOSSES.register_module()
class NormRegLoss(nn.Module):
    def __init__(self, lambda_reg=1.0):  
        super(NormRegLoss, self).__init__()  
        self.lambda_reg = lambda_reg  
  
    def forward(self, x, mask):  
        """  
        计算掩码正态分布正则化损失。  
          
        参数:  
        x (torch.Tensor): 模型的中间输出，形状为 (B, C, T)。  
        mask (torch.Tensor): 掩码，形状为 (B, T)，其中1表示有效位置，0表示无效位置。  
          
        返回:  
        torch.Tensor: 正则化损失（标量）。  
        """  
        # 确保掩码是浮点类型的，以便进行数学运算  
        mask = mask.float()  
          
        # 计算均值和方差，只考虑掩码为1的位置  
        # 使用unsqueeze来扩展掩码的维度，以便与x的维度匹配  
        mask_expanded = mask.unsqueeze(1)  # 形状变为 (B, 1, T)  

        # 计算每个通道在掩码位置上的均值和方差  
        mean = (x * mask_expanded).sum(dim=(0, 2), keepdim=True) / mask_expanded.sum(dim=(0, 2), keepdim=True)  
        var = (x ** 2 * mask_expanded).sum(dim=(0, 2), keepdim=True) / mask_expanded.sum(dim=(0, 2), keepdim=True) - mean ** 2  
          
        # 计算正态分布的正则化损失  
        # 我们希望均值接近0，方差接近1  
        mean_loss = torch.mean((mean - 0) ** 2)  
        var_loss = torch.mean((var - 1) ** 2)  
          
        # 合并损失，并乘以正则化权重  
        reg_loss = self.lambda_reg * (mean_loss + var_loss)  
          
        return reg_loss 
    
@LOSSES.register_module()
class LearnableNormRegLoss(nn.Module):
    def __init__(self, lambda_reg=1.0):  
        super(LearnableNormRegLoss, self).__init__()  
        self.lambda_reg = lambda_reg
        self.learnable_var = nn.Parameter(torch.tensor(1.0)) 
  
    def forward(self, x, mask):  
        """  
        计算掩码正态分布正则化损失。  
          
        参数:  
        x (torch.Tensor): 模型的中间输出，形状为 (B, C, T)。  
        mask (torch.Tensor): 掩码，形状为 (B, T)，其中1表示有效位置，0表示无效位置。  
          
        返回:  
        torch.Tensor: 正则化损失（标量）。  
        """  
        mask = mask.float()  
          
        # 计算均值和方差，只考虑掩码为1的位置  
        # 使用unsqueeze来扩展掩码的维度，以便与x的维度匹配  
        mask_expanded = mask.unsqueeze(1)  # 形状变为 (B, 1, T)  

  
        # 计算每个通道在掩码位置上的均值和方差  
        mean = (x * mask_expanded).sum(dim=(0, 2), keepdim=True) / mask_expanded.sum(dim=(0, 2), keepdim=True)  
        var = (x ** 2 * mask_expanded).sum(dim=(0, 2), keepdim=True) / mask_expanded.sum(dim=(0, 2), keepdim=True) - mean ** 2  
          
        # 计算正态分布的正则化损失  
        # 我们希望均值接近0，方差接近1  
        mean_loss = torch.mean((mean - 0) ** 2)  
        var_loss = torch.mean((var - self.learnable_var) ** 2)  
          
        # 合并损失，并乘以正则化权重    
        reg_loss = self.lambda_reg * (mean_loss + var_loss)  
          
        return reg_loss 