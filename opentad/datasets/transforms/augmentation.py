import torch
import random
import numpy as np
from torch.nn import functional as F


def gaussian_noise(feat, feat_shape, severity=1.0):
    noise = torch.randn(feat_shape).to(feat.device) * 0.001 * severity
    
    return feat + noise

def random_channel_shift(feat, feat_shape, severity=1.0):
    c, t = feat_shape
    selected_num = int(0.01 * severity * c)
    all_channels = list(range(c))
    random.shuffle(all_channels)
    left_shift_idx = torch.tensor(all_channels[:selected_num], dtype=torch.long)
    right_shift_idx = torch.tensor(all_channels[-selected_num:], dtype=torch.long)
    
    left_shift_feat = feat[left_shift_idx] # (selected_num, t)
    right_shift_feat = feat[right_shift_idx] # (selected_num, t)
    
    new_left_shift_feat = torch.zeros_like(left_shift_feat) # (selected_num, t)
    new_left_shift_feat[:, :-1] = left_shift_feat[:, 1:] # (selected_num, t)
    
    new_right_shift_feat = torch.zeros_like(right_shift_feat) # (selected_num, t)
    new_right_shift_feat[:, 1:] = right_shift_feat[:, :-1] # (selected_num, t)
    
    feat[left_shift_idx] = new_left_shift_feat
    feat[right_shift_idx] = new_right_shift_feat
    
    return feat

def temporal_mask(feat, feat_shape, severity=1.0):
    c, t = feat_shape
    mask_rate = 0.01 * severity
    
    return F.dropout1d(feat, p=mask_rate)

def strength_vibration(feat, feat_shape, severity=1.0):
    noise = torch.randn(feat_shape).to(feat.device) * 0.01 * severity
    strength = 1.0 + noise
    
    return feat * strength
