import torch
import torch.nn as nn
import torch.nn.functional as F
from ..bricks import ConvModule, Scale, GradientReversalLayer, CrossAttention, TransformerBlock, MaskMambaBlock


class DomainProjector(nn.Module):
    def __init__(self, feat_dim, ffn_type):
        super(DomainProjector, self).__init__()
        self.conv = ConvModule(feat_dim, 
                               feat_dim, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0, 
                               norm_cfg=dict(type='LN'),
                               act_cfg=dict(type='relu'))
        self.ffn = self.generate_ffn(ffn_type, feat_dim)
        
    def generate_ffn(self, ffn_type, feat_dim):
        if ffn_type == 'ConvModule':
            dom_ffn = ConvModule(
                        feat_dim,
                        feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type="LN"),
                        act_cfg=dict(type="relu"),
                    )
        elif ffn_type == 'Transformer':
            dom_ffn = TransformerBlock(
                                in_channels=feat_dim,
                                n_head=4,
                                n_ds_strides=(1, 1),
                                mha_win_size=19,
                            )
        elif ffn_type == 'Mamba':
            dom_ffn = MaskMambaBlock(
                                n_embd=feat_dim,
                            )
        return dom_ffn
    
    def forward(self, x, mask):
        # x: (B, C, T)
        out = torch.fft.fft(x, dim=-1)
        out = self.conv(out, mask)
        out = F.sigmoid(self.ffn(out, mask)) + out
        out = torch.fft.ifft(out, dim=-1).real
        return out
    
    def forward_pi(self, x, mask):
        out = torch.fft.fft(x, dim=-1)
        out = self.conv(out, mask)
        out = F.sigmoid(self.ffn(out, mask)) + out
        return out
    
    
    
class SODA(nn.Module):
    def __init__(self, model):
        super(SODA, self).__init__()
        self.ffn_type = model.ffn_type
        self.feat_dim = model.projection.in_channels
        self.num_domains = model.rpn_head.num_domains
        self.domain_projector = DomainProjector(self.feat_dim, self.ffn_type)
        self.domain_transfer = CrossAttention(self.feat_dim, n_head=4)
        self.domain_classifier1 = self._create_domain_classifier(self.feat_dim)
        self.domain_classifier2 = self._create_domain_classifier(self.feat_dim)
        
    def _create_domain_classifier(self, feat_dim):
        dom_discriminator = nn.Sequential()
        for i in range(self.num_convs - 1):
            self.dom_discriminator.append(
                ConvModule(
                    feat_dim if i == 0 else 512,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
        dom_discriminator.append(nn.Conv1d(512, self.num_domains, kernel_size=3, padding=1))
        return dom_discriminator
    
    # def forward_train_1(self, inputs, masks, metas, gt_segments, gt_labels, gt_domains, **kwargs):
    #     self.