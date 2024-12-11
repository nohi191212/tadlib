import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .conv import ConvModule
from .transformer import AffineDropPath
from mamba_ssm.modules.mamba_simple import Mamba
from ..builder import MODELS


@MODELS.register_module()
class MaskMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        #use_mamba_type="dbm",
    ):
        super().__init__()
        self.mamba = Mamba(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        # if use_mamba_type == "dbm":
        #     self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        # elif use_mamba_type == "vim":
        #     # vim
        #     self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        # else:
        #     raise NotImplementedError
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1, 2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask


class MaxPooler(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask, **kwargs):
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)

        return out, out_mask.bool()
