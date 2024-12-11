import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
from .conv import ConvModule
from ..builder import MODELS

@MODELS.register_module()
class CrossNegativeAttention(nn.Module):
    """
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # downsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, xkv, xq, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B1, C1, T1 = xkv.size()
        B2, C2, T2 = xq.size()
        C = C1
        B = B1
        assert C1 == C2 == self.n_embd and T1 == T2 and B1 == B2

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(xq, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(xkv, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(xkv, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, None, None, :]), float("-inf"))
        # softmax attn
        att = F.softmax(att, dim=-1) * -1 # <------------------------------ negative attention
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, None, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        out = xkv + out
        
        return out, qx_mask
    
    
@MODELS.register_module()
class CrossAttention(nn.Module):
    """
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # downsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.query_norm = nn.LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.key_norm = nn.LayerNorm(self.n_embd)

        self.value_conv = ConvModule(
            self.n_embd,
            self.n_embd,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=dict(groups=n_embd, bias=False),
        )
        self.value_norm = nn.LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, xkv, xq, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B1, C1, T1 = xkv.size()
        B2, C2, T2 = xq.size()
        C = C1
        B = B1
        assert C1 == C2 == self.n_embd and T1 == T2 and B1 == B2

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(xq, mask)
        q = self.query_norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(xkv, mask)
        k = self.key_norm(k.permute(0, 2, 1)).permute(0, 2, 1)
        v, _ = self.value_conv(xkv, mask)
        v = self.value_norm(v.permute(0, 2, 1)).permute(0, 2, 1)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, None, None, :]), float("-inf"))
        # softmax attn
        att = F.softmax(att, dim=-1) # <------------------------------ negative attention
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, None, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.unsqueeze(1).to(out.dtype)
        out = xkv + out
        
        return out, qx_mask