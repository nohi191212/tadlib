import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pdb

from ..bricks import ConvModule
from ..builder import NECKS

class CrossLayerAggregation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels, 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vk = nn.Conv1d(in_channels, out_channels*2, kernel_size=1)
        self.q = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.scaler = math.sqrt(out_channels)
        
        # upward alignment
        self.psi_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.psi_2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.maxpooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # downward alignment
        self.phi = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, up_input, down_input):
        # target, source = (bsz, c, t)
        up2down = F.interpolate(up_input + self.phi(up_input), size=(down_input.shape[-1]), mode='nearest') # nearest | bilinear
        down2up = self.maxpooling(down_input) + self.psi_2(self.psi_1(down_input))
        vk_u2d = self.vk(up2down)
        vk_d2u = self.vk(down2up)
        v_u2d, k_u2d = torch.split(vk_u2d, [self.out_channels, self.out_channels], dim=1) # (bsz, n_dim, t_len)
        v_d2u, k_d2u = torch.split(vk_d2u, [self.out_channels, self.out_channels], dim=1)
        q_u = self.q(up_input)
        q_d = self.q(down_input)
        
        # attn: channel attention
        attn_d2u = F.softmax(torch.bmm(q_u, k_d2u.permute(0, 2, 1)) / self.scaler, dim=-1)
        attn_u2d = F.softmax(torch.bmm(q_d, k_u2d.permute(0, 2, 1)) / self.scaler, dim=-1)
        up_output = up_input + torch.bmm(attn_d2u, v_d2u) 
        down_output = down_input + torch.bmm(attn_u2d, v_u2d)
        
        return up_output, down_output
        

@NECKS.register_module()
class MSANeck(nn.Module):
    def __init__(
        self,
        in_channels,  # input feature channels, len(in_channels) = #levels
        out_channels,  # output feature channel
        num_levels=0,
        scale_factor=2.0,  # downsampling rate between two fpn levels
        start_level=0,  # start fpn level
        end_level=-1,  # end fpn level
        norm_cfg=dict(type="LN"),  # if no norm, set to none
    ):
        super().__init__()

        self.in_channels = [in_channels] * num_levels
        self.out_channel = out_channels
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(self.in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(self.in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        if norm_cfg is not None:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type
        else:
            self.norm_type = None
            
        # cross-layer aggregation
        self.cla = CrossLayerAggregation(in_channels, out_channels)
            
        # norm
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel

            if self.norm_type == "BN":
                fpn_norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif self.norm_type == "GN":
                fpn_norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif self.norm_type == "LN":
                fpn_norm = nn.LayerNorm(out_channels, eps=1e-6)
            else:
                assert self.norm_type is None
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)
        
        # apply feature aggregation
        ag_inputs = [inputs[lvl] for lvl in range(len(inputs))]
        assert len(ag_inputs) == len(inputs), f"len({ag_inputs}) == len({inputs})"
        for i in range(len(self.fpn_norms)-1, 0, -1):
            x_up = ag_inputs[i + self.start_level]
            x_down = ag_inputs[i-1 + self.start_level]
            ag_inputs[i + self.start_level], ag_inputs[i-1 + self.start_level] = \
                self.cla(x_up, x_down)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = ag_inputs[i + self.start_level]
            if self.norm_type == "LN":
                x = self.fpn_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.fpn_norms[i](x)
            fpn_feats += (x,)
            new_fpn_masks += (fpn_masks[i + self.start_level],)

        return fpn_feats, new_fpn_masks
