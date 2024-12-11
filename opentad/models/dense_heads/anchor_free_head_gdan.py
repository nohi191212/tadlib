import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import (ConvModule, Scale, GradientReversalLayer, CrossNegativeAttention, 
                      TransformerBlock, MaskMambaBlock, DiffTransformerBlock)

@HEADS.register_module()
class AnchorFreeHeadGDAN(nn.Module):
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
        dom_ffn_type='ConvModule', #'Mamba', 'Transformer', 'DiffTransformer'
        cfg=1, # for ablation study
        normreg_loss_weight=1.0,
    ):
        super(AnchorFreeHeadGDAN, self).__init__()

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
        self.dom_ffn_type = dom_ffn_type
        self.num_dom_heads = num_dom_heads
        
        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)
        self.dom_loss = build_loss(loss.dom_loss)
        self.normreg_loss = build_loss(loss.normreg_loss)
        
        # for tSNE
        self.hook_feat = nn.Identity()
        self.hook_label = nn.Identity()
        self.hook_domain = nn.Identity()
        
        # for ablation study
        self.cfg = cfg
        self.normreg_loss_weight = normreg_loss_weight

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_dom_discriminator()
        self._init_heads()
        
        assert self.dom_ffn_type in ['ConvModule', 'Mamba', 'Transformer']
        if self.dom_ffn_type == 'ConvModule':
            self.dom_ffn = ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type="LN"),
                        act_cfg=dict(type="relu"),
                    )
        elif self.dom_ffn_type == 'Transformer':
            self.dom_ffn = TransformerBlock(
                                in_channels=self.feat_channels,
                                n_head=4,
                                n_ds_strides=(1, 1),
                                mha_win_size=19,
                            )
        elif self.dom_ffn_type == 'DiffTransformer':
            raise NotImplementedError
            self.dom_ffn = DiffTransformerBlock(
                                in_channels=self.feat_channels,
                                n_head=4,
                                n_ds_strides=(1, 1),
                                mha_win_size=19,
                            )
        elif self.dom_ffn_type == 'Mamba':
            self.dom_ffn = MaskMambaBlock(
                            n_embd=self.feat_channels,
                            )
        self.cna_cls = CrossNegativeAttention(n_embd=self.feat_channels, n_head=self.num_dom_heads)
        self.cna_reg = CrossNegativeAttention(n_embd=self.feat_channels, n_head=self.num_dom_heads)
        # self.grl_0 = GradientReversalLayer()
        # self.grl_cls = GradientReversalLayer()
        # self.grl_reg = GradientReversalLayer()

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

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, gt_domains, **kwargs):
        cls_pred = []
        reg_pred = []
        dom_pred = []
        
        dom_feats = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            dom_feat, _mask = self.dom_ffn(feat, mask)
            cls_feat, _mask = self.cna_cls(feat, dom_feat, mask)  # 默认配置
            reg_feat, _mask = self.cna_reg(feat, dom_feat, mask)

            dom_feats.append(dom_feat)

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)
        
        # calculate domain prediction
        cat_feats = torch.cat(dom_feats, dim=-1) # B, C, sumT
        avg_feats = []
        for cat_feat in cat_feats:
            avg_feat = torch.mean(cat_feat, dim=1, keepdim=True)
            avg_feats.append(avg_feat)
        avg_feats = torch.stack(avg_feats) # B, C, 1
        dom_pred = self.dom_discriminator(avg_feats).squeeze() # B, num_domains
        
        losses, pos_mask, gt_cls = self.losses(cls_pred, reg_pred, dom_pred, mask_list, points, 
                                               gt_segments, gt_labels, gt_domains,
                                               cat_feats,)
        
        # for tSNE
        features, labels, domains = self.get_pos_data(feat_list, pos_mask, gt_cls, gt_domains)
        features = self.hook_feat(features).detach().cpu()
        labels = self.hook_label(labels).detach().cpu()
        domains = self.hook_domain(domains).detach().cpu()
        
        return losses
    
    

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            dom_feat, _mask = self.dom_ffn(feat, mask)
            cls_feat, _mask = self.cna_cls(feat, dom_feat, mask)
            reg_feat, _mask = self.cna_reg(feat, dom_feat, mask)
            # cls_feat = feat
            # reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores
    
    def get_pos_data(self, feat_list, pos_mask, gt_cls, gt_domains):
        feats = torch.cat(feat_list, dim=-1).transpose(1,2) # B, sumT, C
        labels = torch.argmax(gt_cls, dim=-1) # B, sumT
        domains = torch.cat(gt_domains) # batch_size
        domains = domains.unsqueeze(-1).repeat(1, labels.shape[-1]) # B, sumT
        pos_feats = feats[pos_mask] # N
        pos_labels = labels[pos_mask]
        pos_domains = domains[pos_mask]

        return pos_feats, pos_labels, pos_domains
        

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

    def losses(self, cls_pred, reg_pred, dom_pred, mask_list, points, gt_segments, gt_labels, gt_domains,
               cat_feats=None):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)
        
        gt_dom = torch.cat(gt_domains) # batch_size
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

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / self.num_classes # 源代码这里写错了，原来是 .../(self.num_classes - 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
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
        
        # 4. distribution regularization loss
        cat_masks = torch.cat(mask_list, dim=-1) # B, sumT
        norm_cat_feats = F.layer_norm(cat_feats.permute(0,2,1),cat_feats.permute(0,2,1).shape[-1:], eps=1e-5).permute(0,2,1) # # B, C, sumT
        dist_loss = self.normreg_loss(norm_cat_feats, cat_masks)

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        
        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss * loss_weight,
            "dom_loss": dom_loss * 0.5,
            "dist_loss": dist_loss * self.normreg_loss_weight,
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
class ActionFormerHeadGDAN(AnchorFreeHeadGDAN):
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
        dom_ffn_type='ConvModule',
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
            dom_ffn_type=dom_ffn_type,
            num_dom_heads=num_dom_heads,
            **kwargs,
        )