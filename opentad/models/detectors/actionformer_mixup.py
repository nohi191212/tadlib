import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..bricks import Scale, AffineDropPath

import copy
import random
import numpy as np

@DETECTORS.register_module()
class ActionFormerMixup(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

        n_mha_win_size = self.projection.n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + projection.arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + projection.arch[-1])
            self.mha_win_size = n_mha_win_size
        self.max_seq_len = self.projection.max_seq_len

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert (
                self.max_seq_len % stride == 0
            ), f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks
        elif feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:  # feat_len > self.max_seq_len
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks
    
    def mixup(self, inputs, masks, gt_labels, gt_domains):
        # mixup the input and target cross-domain
        mixed_inputs = copy.deepcopy(inputs) # B, C, T
        assert len(mixed_inputs) == len(gt_domains)
        
        mixup_info = []

        for i in range(len(mixed_inputs)):
            domain_i = gt_domains[i]
            mask_i = masks[i].unsqueeze(0).repeat(inputs[0].shape[0], 1) # C, T
            len_i = masks[i].sum().item()
            if len(gt_labels[i]) == 0:
                mixup_info.append((i, i, 1.0, 0.0, 1.0))
                continue
            label_i = gt_labels[i][0]
            
            j_list = list(range(len(mixed_inputs)))
            j_list.remove(i)
            random.shuffle(j_list)
            
            has_mixup = False
            
            for j in j_list:
                domain_j = gt_domains[j]
                mask_j = masks[j].unsqueeze(0).repeat(inputs[0].shape[0], 1) # C, T
                len_j = masks[j].sum().item()
                if len(gt_labels[j]) == 0:
                    continue
                label_j = gt_labels[j][0]
                if domain_i == domain_j or label_i == label_j:
                    continue
                
                # mixup the input
                lambda_ = torch.rand(1).item()
                lambda_max = max(lambda_, 1 - lambda_)
                lambda_min = min(lambda_, 1 - lambda_)
                lambda_i = np.exp(lambda_max) / (np.exp(lambda_max) + np.exp(lambda_min))
                lambda_j = 1 - lambda_i
                
                # align the length of the input
                new_inputs_j = torch.zeros_like(inputs[i]).unsqueeze(0) # 1, C, T
                valid_part_j = F.interpolate(inputs[j, :, :len_j].unsqueeze(0), # 1, C, T_j
                                             size=len_i, mode='linear', align_corners=False) # 1, C, T_i
                new_inputs_j[:, :, :len_i] = valid_part_j
                ratio = len_i / len_j

                mixed_inputs[i] = inputs[i] * lambda_i + new_inputs_j.squeeze(0) * lambda_j
                
                mixup_info.append((i, j, lambda_i, lambda_j, ratio))
                
                has_mixup = True
                break
            if has_mixup == False:
                mixup_info.append((i, i, 1.0, 0.0, 1.0))
                
        return mixed_inputs, mixup_info
        

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, gt_domains, **kwargs):
        losses = dict()
        
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        # pad the features and unsqueeze the mask for actionformer
        x, masks = self.pad_data(x, masks)
        
        # Mix the input and target cross-domain and category
        mixed_inputs, mixup_info = self.mixup(x, masks, gt_labels, gt_domains)
        x = mixed_inputs
        # import numpy as np
        # if np.unique(torch.cat(gt_domains).cpu().numpy()).shape[0] > 1:
        #     print("这里是actionformer_mixup.py")
        #     import pdb
        #     pdb.set_trace()
        #     quit()
        
        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        loc_losses = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            gt_domains=gt_domains,
            mixup_info=mixup_info,
            **kwargs,
        )
        losses.update(loc_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        predictions = rpn_proposals, rpn_scores
        return predictions

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": cfg["lr"]},
        ]
        return optim_groups
