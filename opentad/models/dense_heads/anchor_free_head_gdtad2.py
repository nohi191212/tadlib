import math
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d  
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from collections import deque
import random

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import (ConvModule, Scale, GradientReversalLayer, CrossNegativeAttention, CrossAttention,
                      TransformerBlock, MaskMambaBlock, DiffTransformerBlock)

def channel_norm(x):
    # x: B, C, T
    x_norm = F.layer_norm(x.permute(0,2,1),x.permute(0,2,1).shape[-1:], eps=1e-5).permute(0,2,1)
    return x_norm

@HEADS.register_module()
class AnchorFreeHeadGDTAD2(nn.Module):
    def __init__(
        self,
        num_classes,
        num_domains,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        filter_similar_gt=True,
        num_dom_heads=16,
        dom_proj_type='ConvModule', #'Mamba', 'Transformer', 'DiffTransformer'
        cfg=1, # for ablation study
        normreg_loss_weight=0.1,
        stage_2_loss_weights=[0.5, 0.5, 1.0, 1.0],
        gen_strength=2.0,
    ):
        super(AnchorFreeHeadGDTAD2, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self.num_domains = num_domains        
        self.dom_proj_type = dom_proj_type
        self.num_dom_heads = num_dom_heads
        
        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)
        self.dom_loss = build_loss(loss.dom_loss)
        self.normreg_loss = build_loss(loss.normreg_loss)
        self.gen_dom_loss = build_loss(loss.gen_dom_loss)
        self.cls_da_loss = build_loss(loss.cls_da_loss)
        self.reg_da_loss = build_loss(loss.reg_da_loss)
        self.feat_da_loss = build_loss(loss.feat_da_loss)
        
        # domain feature queue for domain generation
        self.queue = deque(maxlen=100)
        
        self.stage_2_loss_weights = stage_2_loss_weights
        
        # for tSNE
        self.hook_feat = nn.Identity()
        self.hook_label = nn.Identity()
        self.hook_domain = nn.Identity()
        self.hook_dom_shift = nn.Identity()
        self.hook_recons_shift = nn.Identity()
        self.hook_gen_shift = nn.Identity()
        self.hook_dom_feat = nn.Identity()
        self.hook_gen_dom_feat = nn.Identity()
        
        # for ablation study
        self.cfg = cfg
        self.normreg_loss_weight = normreg_loss_weight
        self.gen_strength = gen_strength

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_dom_discriminator()
        self._init_heads()
        
        assert self.dom_proj_type in ['ConvModule', 'Mamba', 'Transformer']
        if self.dom_proj_type == 'ConvModule':
            self.dom_proj = ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type="LN"),
                        act_cfg=dict(type="relu"),
                    )
        elif self.dom_proj_type == 'Transformer':
            self.dom_proj = TransformerBlock(
                                in_channels=self.feat_channels,
                                n_head=4,
                                n_ds_strides=(1, 1),
                                mha_win_size=19,
                            )
        elif self.dom_proj_type == 'DiffTransformer':
            raise NotImplementedError
            self.dom_proj = DiffTransformerBlock(
                                in_channels=self.feat_channels,
                                n_head=4,
                                n_ds_strides=(1, 1),
                                mha_win_size=19,
                            )
        elif self.dom_proj_type == 'Mamba':
            self.dom_proj = MaskMambaBlock(
                            n_embd=self.feat_channels,
                            )
        self.cna_cls = CrossNegativeAttention(n_embd=self.feat_channels, n_head=self.num_dom_heads)
        self.cna_reg = CrossNegativeAttention(n_embd=self.feat_channels, n_head=self.num_dom_heads)
        
        self.domain_generator_cls = deepcopy(self.dom_proj)
        self.domain_generator_reg = deepcopy(self.dom_proj)
        
        self.stop_gradient = GradientReversalLayer(lambda_=0.0)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
            
    def _init_dom_discriminator(self):
        self.dom_discriminator = nn.Sequential()
        for i in range(self.num_convs - 1):
            self.dom_discriminator.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
        self.dom_discriminator.append(nn.Conv1d(self.feat_channels, self.num_domains, kernel_size=3, padding=1))

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)
            nn.init.constant_(self.dom_discriminator[-1].bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, gt_domains, 
                      stage=1, t_cls_pred=None, t_reg_pred=None, gen_dom_shifts_cls=None, gen_dom_shifts_reg=None,
                      **kwargs):
        if stage == 1:
            cls_pred = []
            reg_pred = []
            dom_pred = []
            
            dom_shifts_cls = []
            recons_shifts_cls = []
            
            dom_shifts_reg = []
            recons_shifts_reg = []
            
            dom_feats = []

            for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
                dom_feat, _mask = self.dom_proj(feat, mask)
                nodom_feat_cls, _mask = self.cna_cls(feat, dom_feat, mask)
                nodom_feat_reg, _mask = self.cna_reg(feat, dom_feat, mask)
                cls_feat = nodom_feat_cls
                reg_feat = nodom_feat_reg
                
                dom_feats.append(dom_feat)
                
                for i in range(self.num_convs):
                    cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                    reg_feat, mask = self.reg_convs[i](reg_feat, mask)
                
                cls_pred.append(self.cls_head(cls_feat))
                reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))
                
                # for domain shift generator
                dom_shift_cls = self.stop_gradient((feat - nodom_feat_cls).detach())
                dom_shift_reg = self.stop_gradient((feat - nodom_feat_reg).detach())
                #norm_dom_feat = channel_norm(dom_feat.detach()) # normalized
                # [domain generation loss] only train the [domain shift generator], so detach the dom_feat
                recons_shift_cls, _mask = self.domain_generator_cls(dom_feat.detach(), mask.detach())
                recons_shift_reg, _mask = self.domain_generator_reg(dom_feat.detach(), mask.detach())
                
                dom_shifts_cls.append(dom_shift_cls)
                dom_shifts_reg.append(dom_shift_reg)
                recons_shifts_cls.append(recons_shift_cls)
                recons_shifts_reg.append(recons_shift_reg)

            points = self.prior_generator(feat_list)
            
            # calculate domain prediction (list 6)
            avg_feats = [] # L, B, C
            for _dom_feat, mask in zip(dom_feats, mask_list): # repeat L times
                _dom_feat = _dom_feat.permute(0,2,1) # B, T, C
                avg_feats_l = []
                for _v_dom_feat, _v_mask in zip(_dom_feat, mask): # repeat B times
                    _v_dom_feat = _v_dom_feat[_v_mask] # T, C
                    avg_feat = torch.mean(_v_dom_feat, dim=0, keepdim=False) # C
                    avg_feats_l.append(avg_feat)
                avg_feats_l = torch.stack(avg_feats_l) # B, C
                avg_feats.append(avg_feats_l)
            avg_feats = torch.stack(avg_feats).mean(dim=0, keepdim=False).unsqueeze(-1) # B, C, 1
            dom_pred = self.dom_discriminator(avg_feats).squeeze() # B, num_domains
            
            # generation candidate
            normed_avg_feats = avg_feats.detach() # B, C, 1
            for avg_feat in normed_avg_feats:
                # if len(self.queue) == self.queue.maxlen:
                #     del self.queue[random.randint(0, len(self.queue)-1)]
                #     self.queue.append(deepcopy(avg_feat.detach())) # max_qsize, C, 1
                # else:
                self.queue.append(deepcopy(avg_feat.detach())) # max_qsize, C, 1
            
            # for norm reg loss
            cat_feats = torch.cat(dom_feats, dim=-1) # B, C, sumT
            
            # calculate losses
            losses, pos_mask, gt_cls = self.losses_stage1(cls_pred, reg_pred, dom_pred, mask_list, points, 
                                            gt_segments, gt_labels, gt_domains, # cls_loss, reg_loss, dom_loss
                                            cat_feats, # norm_reg_loss
                                            recons_shifts_cls, recons_shifts_reg, dom_shifts_cls, dom_shifts_reg, # gen_dom_loss
                                        )
            
            # for tSNE, dom_shifts是学习自然得到的偏移，recons_shifts是生成的偏移
            features, labels, domains, pos_dom_shifts, pos_recons_shifts = \
                self.get_pos_data(feat_list, pos_mask, gt_cls, gt_domains, dom_shifts_cls, recons_shifts=recons_shifts_cls)
            features = self.hook_feat(features).detach().cpu()
            labels = self.hook_label(labels).detach().cpu()
            domains = self.hook_domain(domains).detach().cpu()
            pos_dom_shifts = self.hook_dom_shift(pos_dom_shifts).detach().cpu()
            pos_recons_shifts = self.hook_recons_shift(pos_recons_shifts).detach().cpu()
            
            return losses
        
        elif stage == 2:
            cls_pred = []
            reg_pred = []
            
            aug_cls_pred = []
            aug_reg_pred = []
            
            nodom_feats_cls = []
            nodom_feats_reg = []
            aug_nodom_feats_cls = []
            aug_nodom_feats_reg = []
            
            dom_shifts_cls = []
            dom_shifts_reg = []
            
            for l, (feat, mask, gen_dom_shift_cls, gen_dom_shift_reg) in enumerate(zip(feat_list, mask_list, 
                                                                gen_dom_shifts_cls, gen_dom_shifts_reg)):
                dom_feat, _mask = self.dom_proj(feat, mask)
                nodom_feat_cls, _mask = self.cna_cls(feat, dom_feat, mask)
                nodom_feat_reg, _mask = self.cna_reg(feat, dom_feat, mask)
                cls_feat = nodom_feat_cls
                reg_feat = nodom_feat_reg
                
                dom_shift_cls = self.stop_gradient((feat - nodom_feat_cls).detach())
                dom_shift_reg = self.stop_gradient((feat - nodom_feat_reg).detach())
                
                aug_feat_cls = feat + gen_dom_shift_reg
                aug_feat_reg = feat + gen_dom_shift_reg
                aug_dom_feat_cls, _mask = self.dom_proj(aug_feat_cls, mask)
                aug_dom_feat_reg, _mask = self.dom_proj(aug_feat_reg, mask)
                aug_nodom_feat_cls, _mask = self.cna_cls(aug_feat_cls, aug_dom_feat_cls, mask)
                aug_nodom_feat_reg, _mask = self.cna_reg(aug_feat_reg, aug_dom_feat_reg, mask)
                aug_cls_feat = aug_nodom_feat_cls
                aug_reg_feat = aug_nodom_feat_reg
                
                for i in range(self.num_convs):
                    cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                    reg_feat, mask = self.reg_convs[i](reg_feat, mask)
                    aug_cls_feat, mask = self.cls_convs[i](aug_cls_feat, mask)
                    aug_reg_feat, mask = self.reg_convs[i](aug_reg_feat, mask)

                cls_pred.append(self.cls_head(cls_feat))
                reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))
                aug_cls_pred.append(self.cls_head(aug_cls_feat))
                aug_reg_pred.append(F.relu(self.scale[l](self.reg_head(aug_reg_feat))))
                
                dom_shifts_cls.append(dom_shift_cls)
                dom_shifts_reg.append(dom_shift_reg)
                nodom_feats_cls.append(nodom_feat_cls)
                nodom_feats_reg.append(nodom_feat_reg)
                aug_nodom_feats_cls.append(aug_nodom_feat_cls)
                aug_nodom_feats_reg.append(aug_nodom_feat_reg)

            points = self.prior_generator(feat_list)
            
            # calculate losses
            losses, pos_mask, gt_cls = self.losses_stage2(cls_pred, reg_pred, mask_list, points, 
                                            gt_segments, gt_labels, # cls_loss, reg_loss, 
                                            t_cls_pred, t_reg_pred, # distill_loss
                                            aug_cls_pred, aug_reg_pred, # domain_agnostic_consistency_loss
                                            nodom_feats_cls, aug_nodom_feats_cls,
                                             nodom_feats_reg, aug_nodom_feats_reg,
                                        )
            
            # for tSNE
            features, labels, domains, pos_dom_shifts, pos_gen_shifts = \
                self.get_pos_data(feat_list, pos_mask, gt_cls, gt_domains, dom_shifts_cls, gen_shifts=gen_dom_shifts_cls)
            features = self.hook_feat(features).detach().cpu()
            labels = self.hook_label(labels).detach().cpu()
            domains = self.hook_domain(domains).detach().cpu()
            pos_dom_shifts = self.hook_dom_shift(pos_dom_shifts).detach().cpu()
            pos_gen_shifts = self.hook_gen_shift(pos_gen_shifts).detach().cpu()
            
            return losses
    
    def forward_distill(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):        
        cls_pred = []
        reg_pred = []
        dom_shifts_cls = []
        dom_shifts_reg = []
        
        gen_dom_feats = []
        dom_feats = []
        
        B, C, _ = feat_list[0].shape
        queue_len = len(self.queue)
        if queue_len != 0:
            gaussian_seed = (torch.randn(B, queue_len) * self.gen_strength + 1/queue_len)[..., None, None].to(feat_list[0].device) # [B, Q, 1, 1]
            all_queue_domain = torch.stack(list(self.queue)) # Q, C, 1
            all_queue_domain.unsqueeze_(0).repeat(B, 1, 1, 1) # B, Q, C, 1
            generated_domain = (all_queue_domain * gaussian_seed).mean(dim=1, keepdim=False) # B, C, 1
        else:
            generated_domain = torch.zeros(B, C, 1).to(feat_list[0].device)
        
        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            dom_feat, _mask = self.dom_proj(feat, mask)
            nodom_feat_cls, _mask = self.cna_cls(feat, dom_feat, mask)
            nodom_feat_reg, _mask = self.cna_reg(feat, dom_feat, mask)
            cls_feat = nodom_feat_cls
            reg_feat = nodom_feat_reg
            
            dom_feats.append(dom_feat)
            
            gen_domain_l = generated_domain.repeat(1, 1, cls_feat.shape[-1]) # B, C, T
            gen_domain_l += torch.randn_like(gen_domain_l)  # add noise
            gen_domain_l = gaussian_filter1d(gen_domain_l.cpu().detach().numpy(), sigma=self.gen_strength*0.5, axis=-1)
            gen_domain_l = torch.from_numpy(gen_domain_l).to(cls_feat.device)
            gen_dom_feats.append(gen_domain_l)
            
            dom_shift_cls, _mask = self.domain_generator_cls(gen_domain_l, mask)
            dom_shift_reg, _mask = self.domain_generator_reg(gen_domain_l, mask)

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))
            dom_shifts_cls.append(dom_shift_cls)
            dom_shifts_reg.append(dom_shift_reg)
        
        # store the latest domain feature for domain shift generator
        avg_feats = [] # L, B, C
        for _dom_feat, mask in zip(dom_feats, mask_list): # repeat L times
            _dom_feat = _dom_feat.permute(0,2,1) # B, T, C
            avg_feats_l = []
            for _v_dom_feat, _v_mask in zip(_dom_feat, mask): # repeat B times
                _v_dom_feat = _v_dom_feat[_v_mask] # T, C
                avg_feat = torch.mean(_v_dom_feat, dim=0, keepdim=False) # C
                avg_feats_l.append(avg_feat)
            avg_feats_l = torch.stack(avg_feats_l) # B, C
            avg_feats.append(avg_feats_l)
        avg_feats = torch.stack(avg_feats).mean(dim=0, keepdim=False).unsqueeze(-1) # B, C, 1
        
        normed_avg_feats = avg_feats.detach() # B, C, 1
        for avg_feat in normed_avg_feats:
            # if len(self.queue) == self.queue.maxlen:
            #     del self.queue[random.randint(0, len(self.queue)-1)]
            #     self.queue.append(deepcopy(avg_feat.detach())) # max_qsize, C, 1
            # else:
            self.queue.append(deepcopy(avg_feat.detach())) # max_qsize, C, 1
            
        points = self.prior_generator(feat_list)
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)
        
        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
            
        _features, _labels, _domains, pos_dom_feats, pos_gen_dom_feats = \
            self.get_pos_data(feat_list, pos_mask, gt_cls, gt_domains=None, 
                              dom_feats=dom_feats, gen_dom_feats=gen_dom_feats)
        pos_dom_feats = self.hook_dom_feat(pos_dom_feats).detach().cpu()
        pos_gen_dom_feats = self.hook_gen_dom_feat(pos_gen_dom_feats).detach().cpu()

        return cls_pred, reg_pred, dom_shifts_cls, dom_shifts_reg
        

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            dom_feat, _mask = self.dom_proj(feat, mask)
            nodom_feat_cls, _mask = self.cna_cls(feat, dom_feat, mask)
            nodom_feat_reg, _mask = self.cna_reg(feat, dom_feat, mask)
            cls_feat = nodom_feat_cls
            reg_feat = nodom_feat_reg

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores
    
    def get_pos_data(self, feat_list, pos_mask, gt_cls, gt_domains, 
                     dom_shifts=None, recons_shifts=None, gen_shifts=None, dom_feats=None, gen_dom_feats=None):
        feats = torch.cat(feat_list, dim=-1).transpose(1,2) # B, sumT, C
        labels = torch.argmax(gt_cls, dim=-1) # B, sumT
        pos_feats = feats[pos_mask] # N
        pos_labels = labels[pos_mask]
        
        if gt_domains:
            domains = torch.cat(gt_domains) # batch_size
            domains = domains.unsqueeze(-1).repeat(1, labels.shape[-1]) # B, sumT
            pos_domains = domains[pos_mask]
        else:
            pos_domains = None
        
        if recons_shifts is not None:
            pos_dom_shifts = torch.cat(dom_shifts, dim=-1).transpose(1,2)[pos_mask]
            pos_recons_shifts = torch.cat(recons_shifts, dim=-1).transpose(1,2)[pos_mask]
            return pos_feats, pos_labels, pos_domains, pos_dom_shifts, pos_recons_shifts
        
        if gen_shifts is not None:
            pos_dom_shifts = torch.cat(dom_shifts, dim=-1).transpose(1,2)[pos_mask]
            pos_gen_shifts = torch.cat(gen_shifts, dim=-1).transpose(1,2)[pos_mask]
            return pos_feats, pos_labels, pos_domains, pos_dom_shifts, pos_gen_shifts
        
        if dom_feats is not None:
            pos_dom_feats = torch.cat(dom_feats, dim=-1).transpose(1,2)[pos_mask]
            pos_gen_dom_feats = torch.cat(gen_dom_feats, dim=-1).transpose(1,2)[pos_mask]
            return pos_feats, pos_labels, pos_domains, pos_dom_feats, pos_gen_dom_feats

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)  # [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        # apply regression to get refined proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        # proposal scores
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]

        # mask out invalid, and return a list with batch size
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,num_classes]
        return new_proposals, new_scores

    def losses_stage1(
        self, 
        cls_pred, reg_pred, dom_pred, mask_list, points, 
        gt_segments, # reg_loss
        gt_labels, # cls_loss 
        gt_domains, # dom_loss
        cat_feats, # norm_reg_loss
        recons_shifts_cls,
        recons_shifts_reg,
        dom_shifts_cls,
        dom_shifts_reg,
    ):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)
        
        gt_dom = torch.cat(gt_domains) # batch_size
        batch_size = gt_dom.shape[0]
        gt_dom = F.one_hot(gt_dom.long(), self.num_domains).to(dom_pred.dtype) # batch_size, num_domains
        if len(dom_pred.shape) == 1:
            dom_pred = dom_pred.unsqueeze(0)
        
        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss (GT)
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / self.num_classes # 源代码这里写错了，原来是 .../(self.num_classes - 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples) (GT)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer
            
        # 3. domain loss
        dom_loss = self.dom_loss(dom_pred, gt_dom, reduction="sum")
        dom_loss /= loss_normalizer
        
        # 4. normal regularization loss
        cat_masks = torch.cat(mask_list, dim=-1) # B, sumT
        norm_cat_feats = channel_norm(cat_feats)
        dist_loss = self.normreg_loss(norm_cat_feats, cat_masks)
        
        # 5. domain generation loss
        recons_shifts_cls = torch.cat(recons_shifts_cls, dim=-1).permute(0, 2, 1)[valid_mask.detach()]
        dom_shifts_cls = torch.cat(dom_shifts_cls, dim=-1).permute(0, 2, 1)[valid_mask.detach()]
        gen_dom_loss_cls = self.gen_dom_loss(recons_shifts_cls, dom_shifts_cls)
        gen_dom_loss_cls /= loss_normalizer
        
        recons_shifts_reg = torch.cat(recons_shifts_reg, dim=-1).permute(0, 2, 1)[valid_mask.detach()]
        dom_shifts_reg = torch.cat(dom_shifts_reg, dim=-1).permute(0, 2, 1)[valid_mask.detach()]
        gen_dom_loss_reg = self.gen_dom_loss(recons_shifts_reg, dom_shifts_reg)
        gen_dom_loss_reg /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        
        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss * loss_weight,
            "dom_loss": dom_loss * 0.5,
            "dist_loss": dist_loss * self.normreg_loss_weight,
            "gen_dom_loss_cls": gen_dom_loss_cls,
            "gen_dom_loss_reg": gen_dom_loss_reg,
        }

        return losses, pos_mask, gt_cls
    
    
    def losses_stage2(
        self, 
        cls_pred, reg_pred, mask_list, points, 
        gt_segments, # reg_loss
        gt_labels, # cls_loss 
        t_cls_pred, # distill_cls_loss
        t_reg_pred, # distill_reg_loss
        aug_cls_pred, # da_cls_loss
        aug_reg_pred, # da_reg_loss
        nodom_feats_cls, 
        aug_nodom_feats_cls,
        nodom_feats_reg, 
        aug_nodom_feats_reg,
    ):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)
        
        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        neg_mask = torch.logical_and((gt_cls.sum(-1) == 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss (GT)
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / self.num_classes # 源代码这里写错了，原来是 .../(self.num_classes - 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples) (GT)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer
            
        # domain-agnostic loss
        nodom_feats_cls = [x.permute(0, 2, 1) for x in nodom_feats_cls]
        nodom_feats_cls = torch.cat(nodom_feats_cls, dim=1)
        aug_nodom_feats_cls = [x.permute(0, 2, 1) for x in aug_nodom_feats_cls] # B, sumT, C
        aug_nodom_feats_cls = torch.cat(aug_nodom_feats_cls, dim=1)
        neg_da_loss_cls = self.feat_da_loss(nodom_feats_cls[neg_mask], aug_nodom_feats_cls[neg_mask]) * 0.1
        pos_da_loss_cls = self.feat_da_loss(nodom_feats_cls[pos_mask], aug_nodom_feats_cls[pos_mask])
        
        nodom_feats_reg = [x.permute(0, 2, 1) for x in nodom_feats_reg]
        nodom_feats_reg = torch.cat(nodom_feats_reg, dim=1)
        aug_nodom_feats_reg = [x.permute(0, 2, 1) for x in aug_nodom_feats_reg] # B, sumT, C
        aug_nodom_feats_reg = torch.cat(aug_nodom_feats_reg, dim=1)
        neg_da_loss_reg = self.feat_da_loss(nodom_feats_reg[neg_mask], aug_nodom_feats_reg[neg_mask]) * 0.1
        pos_da_loss_reg = self.feat_da_loss(nodom_feats_reg[pos_mask], aug_nodom_feats_reg[pos_mask])
        
        da_loss = (neg_da_loss_cls + pos_da_loss_cls + neg_da_loss_reg + pos_da_loss_reg) * 0.5
        
        # 3. classification consistency loss against generated domain shift
        # now: cls_pred: N, num_classes
        aug_cls_pred = [x.permute(0, 2, 1) for x in aug_cls_pred]
        aug_cls_pred = torch.cat(aug_cls_pred, dim=1)[valid_mask]
        cls_cons_loss = self.cls_da_loss(aug_cls_pred, cls_pred.detach()) # JSDivergence Loss
        
        # 4. regression consistency loss against generated domain shift
        # pred_segments: N_POS, 2
        aug_pred_segments = self.get_refined_proposals(points, aug_reg_pred)[pos_mask]
        reg_cons_loss = self.reg_da_loss(aug_pred_segments, pred_segments.detach(), reduction="sum")
        reg_cons_loss /= loss_normalizer
        
        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        
        w = self.stage_2_loss_weights
        losses = {
            "cls_loss": cls_loss * w[0],
            "reg_loss": reg_loss * loss_weight * w[1],
            "cls_cons_loss": cls_cons_loss * w[2],
            "reg_cons_loss": reg_cons_loss * w[3],
            "da_loss": da_loss,
        }

        return losses, pos_mask, gt_cls
    
    

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []
        
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]
            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)
            
            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg



@HEADS.register_module()
class ActionFormerHeadGDTAD2(AnchorFreeHeadGDTAD2):
    def __init__(
        self,
        num_classes,
        num_domains,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        dom_proj_type='ConvModule',
        num_dom_heads=16,
        **kwargs,
    ):
        super().__init__(
            num_classes,
            num_domains,
            in_channels,
            feat_channels,
            num_convs=num_convs,
            cls_prior_prob=cls_prior_prob,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            loss_weight=loss_weight,
            label_smoothing=label_smoothing,
            center_sample=center_sample,
            center_sample_radius=center_sample_radius,
            dom_proj_type=dom_proj_type,
            num_dom_heads=num_dom_heads,
            **kwargs,
        )