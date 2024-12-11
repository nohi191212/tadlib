# Adapt from: https://github.com/WoojuLee24/OA-DG/blob/main/mmdet/datasets/pipelines/oa_mix.py

import torch
import torch.nn.functional as F
import torchvision
import scipy
import numpy as np
from collections.abc import Sequence
from einops import rearrange, reduce
import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .augmentation import gaussian_noise, random_channel_shift, temporal_mask, strength_vibration
from ..builder import PIPELINES

def get_aug_list():
    aug_list = [gaussian_noise, random_channel_shift, temporal_mask, strength_vibration]
    return aug_list

def bbox_overlaps(segs_1, segs_2):
    # segs_1: [1,2] torch.Tensor
    # segs_2: [N,2] torch.Tensor
    if len(segs_2) == 0:
        return torch.tensor([0.])
    segs_1 = segs_1.repeat(len(segs_2), 1) # [N, 2]
    min_begin = torch.min(segs_1[:, 0], segs_2[:, 0])
    max_begin = torch.max(segs_1[:, 0], segs_2[:, 0])
    min_end = torch.min(segs_1[:, 1], segs_2[:, 1])
    max_end = torch.max(segs_1[:, 1], segs_2[:, 1])

    inter = min_end - max_begin
    union = max_end - min_begin
    ious = inter / union
    
    return ious

@PIPELINES.register_module()
class OAMix1D:
    def __init__(
        self,
        # settings for augmentation
        num_views=2,severity=10,
        mixture_width=3, mixture_depth=-1,   # Mixing strategy (AugMix setting)
        random_seg_scale=(0.05, 0.2), # multi-level transformation
        oa_random_seg_scale=(0.05, 0.2), # object-aware mixing
        spatial_ratio=4, sigma_ratio=0.3,  # Smoothing strategy to improve speed
    ):
        super(OAMix1D, self).__init__()        
        self.aug_list = get_aug_list()
        
        # follow AugMix settings
        self.num_views = num_views  # number of augmented views
        self.severity = severity  # strength of transformation (0~10)
        self.aug_prob_coeff = 1.0
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        
        """ Multi-level transformation """
        self.random_seg_scale = random_seg_scale

        """ Object-aware mixing """
        self.oa_random_seg_scale = oa_random_seg_scale
        
        self.score_thresh = 2.0
        
        # Smoothing strategy (for fg and bg)
        self.spatial_ratio = spatial_ratio
        self.sigma_ratio = sigma_ratio
        
        self._history = {}
    
    # customed saliency calculation
    def get_saliency_map(self, feat, gt_segments):
        c, t = feat.shape
        gt_mask = torch.zeros(t, dtype=torch.float32).to(feat.device)
        for seg in gt_segments:
            t1, t2 = seg
            t1 = np.floor(t1).int()
            t2 = np.ceil(t2).int()
            gt_mask[t1:t2] = 1.0
        saliency_map = gaussian_filter1d(gt_mask.cpu().numpy(), sigma=0.2, axis=0) # T
        saliency_map = torch.from_numpy(saliency_map).to(feat.device) # T

        return saliency_map
    
    @staticmethod
    def _get_mask(seg, feat_shape):
        t1, t2 = seg
        t1 = np.floor(t1).int()
        t2 = np.ceil(t2).int()
        mask = torch.zeros(feat_shape, dtype=torch.bool)            
        mask[:, t1:t2] = True
        return mask
    
    def get_fg_segments(self, feat, gt_segments):
        if hasattr(self._history, "fg_seg_list"):
            return self._history["fg_seg_list"], self._history["fg_mask_list"], self._history["fg_score_list"]
        else:
            fg_seg_list, fg_mask_list, fg_score_list = gt_segments, [], []
            for i, gt_seg in enumerate(gt_segments):
                """ Object-aware mixing: compute saliency score for each fg region """
                t1, t2 = np.array(gt_seg, dtype=np.int32)
                if t2 - t1 < self.spatial_ratio:
                    # If it is too small, the score will be -1.
                    fg_score_list.append(-1)
                else:
                    saliency_score = self.saliency_map[t1:t2].sum()
                    fg_score_list.append(saliency_score)

                # Hacks to speed up blurred mask generation
                fg_mask = self._get_mask(gt_seg, feat.shape)
                fg_mask_list.append(fg_mask)

            self._history.update({"fg_seg_list": fg_seg_list})
            self._history.update({"fg_mask_list": fg_mask_list})
            self._history.update({"fg_score_list": fg_score_list})
            return fg_seg_list, fg_mask_list, fg_score_list
    
    def get_random_segments(self, feat, scale, 
                            num_segs, return_score=False, fg_seg_list=None, fg_score_list=None,
                            max_iters=50, eps=1e-6):
        if return_score:
            assert fg_seg_list is not None and fg_score_list is not None
        feat_len = feat.shape[1]
        
        random_seg_list, random_mask_list, random_score_list = [], [], []

        # Randomly determines the number of boxes to be generated
        target_num_bboxes = np.random.randint(*num_segs) if isinstance(num_segs, tuple) else num_segs
        for i in range(max_iters):
            # Stop random region generation when the determined number of boxes has been generated.
            if len(random_mask_list) >= target_num_bboxes:
                break

            # Generate a random bbox.
            t1 = np.random.randint(0, feat_len)
            _scale = np.random.uniform(*scale) * feat_len # length of the random region
            seg_len = int(_scale)

            if t1 + seg_len > feat_len:
                continue # incorrectly generated segment

            t2 = t1 + seg_len
            random_seg = torch.tensor([[t1, t2]], dtype=torch.int).to(feat.device)

            # Except if it overlaps with existing boxes
            
            if len(random_seg_list) == 0:
                ious = bbox_overlaps(random_seg, torch.tensor(random_seg_list).to(feat.device))
            else:
                ious = bbox_overlaps(random_seg, torch.stack(random_seg_list, dim=0).to(feat.device))

            if ious.sum() > eps:
                continue

            # compute saliency score for each fg region
            if return_score:
                ious = bbox_overlaps(random_seg, fg_seg_list)

                final_score = float("inf")
                if ious.sum() > eps:
                    # If the random seg is overlapped with fg region
                    for i, (iou, fg_seg, fg_score) in enumerate(zip(ious, fg_seg_list, fg_score_list)):
                        # If there is no overlapping area, it is not reflected in the saliency score.
                        t1_fg, t2_fg = fg_seg
                        if iou == 0.0 or abs(t1_fg - t2_fg) <= 1:
                            continue
                        if fg_score < final_score:
                            final_score = fg_score
                random_score_list.append(final_score)

            # Generate the mask of the random segment.
            random_mask = self._get_mask(random_seg[0], feat.shape)
            random_mask_list.append(random_mask)
            random_seg_list += list(random_seg)

        if return_score:
            return random_seg_list, random_mask_list, random_score_list
        else:
            return random_seg_list, random_mask_list

    def __call__(self, data):
        # data: dict{'inputs', 'masks', 'gt_segments', 'gt_labels', 'metas', ...}
        feature = data["inputs"]  # [C,T]
        mask = data['masks'] # T
        gt_segments = data["gt_segments"]  # [N,2]
        
        self._history = {}
        self.saliency_map = self.get_saliency_map(feature[:, mask].clone(), gt_segments.clone())
        feat_oamix = torch.zeros_like(feature).to(feature.device)
        feat_oamix[:, mask] = self.oamix(feat=feature[:, mask].clone(), gt_segments=gt_segments.clone())
        data["inputs"] = feat_oamix

        return data
    
    def oamix(self, feat, gt_segments):
        feat_len = feat.shape[1]

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        
        # Multi-level transformation: get_random_segments() & get_masks()
        random_seg_list, random_mask_list = self.get_random_segments(
            feat, self.random_seg_scale, num_segs=(1, 3))
        self._history.update({"random_seg_list": np.stack(random_seg_list, axis=0)})
        fg_seg_list, fg_mask_list, fg_score_list = self.get_fg_segments(feat=feat, gt_segments=gt_segments)
        
        # Initialize I_oamix with zeros
        feat_oamix = torch.zeros_like(feat, dtype=torch.float32).to(feat.device)
        for i in range(self.mixture_width):
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            feat_aug = copy.deepcopy(feat) # [C,T]
            for _ in range(depth):
                """ Multi-level transformation """
                # Initialize I_aug with zeros
                feat_tmp = torch.zeros_like(feat, dtype=torch.float32).to(feat.device)
                for _randbox, _randmask in zip(random_seg_list, random_mask_list):
                    feat_tmp += _randmask * self.aug(feat_aug, feat.shape, severity=self.severity)

                union_mask = torch.zeros_like(random_mask_list[0])
                for rand_mask in random_mask_list:
                    union_mask = union_mask | rand_mask
                
                feat_aug = feat_tmp + (1.0 - union_mask.float()) * self.aug(feat_aug, feat.shape, severity=self.severity)

            feat_oamix += ws[i] * feat_aug

        """ Object-aware mixing """
        oa_target_seg_list, oa_target_mask_list, oa_target_score_list = self.get_regions_for_object_aware_mixing(
            feat, fg_seg_list, fg_mask_list, fg_score_list)
        feat_oamix = self.object_aware_mixing(feat, feat_oamix, oa_target_mask_list, oa_target_score_list)
        
        return feat_oamix
    
    def get_regions_for_object_aware_mixing(self, feat, fg_seg_list, fg_mask_list, fg_score_list):
        oa_target_seg_list, oa_target_mask_list, oa_target_score_list = [], [], []
        for idx, (seg, mask, score) in enumerate(zip(fg_seg_list, fg_mask_list, fg_score_list)):
            # For the objects with low saliency score,
            if score <= self.score_thresh:
                oa_target_seg_list.append(seg)
                oa_target_mask_list.append(mask)
                oa_target_score_list.append(score)
        oa_random_seg_list, oa_random_mask_list, oa_random_score_list = self.get_random_segments(
            feat, self.oa_random_seg_scale,
            num_segs=min(max(len(oa_target_seg_list), 1), 5),
            return_score=True, fg_seg_list=fg_seg_list, fg_score_list=fg_score_list
        )
        oa_target_seg_list += oa_random_seg_list
        oa_target_mask_list += oa_random_mask_list
        oa_target_score_list += oa_random_score_list
        self._history.update({"oa_random_seg_list": oa_random_seg_list})
        return oa_random_seg_list, oa_target_mask_list, oa_target_score_list
    
    def aug(self, feat, feat_shape, severity=10):
        aug_func = np.random.choice(self.aug_list)
        return aug_func(feat, feat_shape, severity=severity)
    
    def object_aware_mixing(self, feat, feat_aug, mask_list, score_list):
        m = np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff)

        orig, aug = torch.zeros_like(feat).to(feat.device), torch.zeros_like(feat).to(feat.device)
        mask_sum = torch.zeros_like(feat, dtype=torch.float).to(feat.device)
        mask_max_list = []
        for i, (mask, score) in enumerate(zip(mask_list, score_list)):
            mask = mask.float()
            # Get union of masks
            mask_sum += mask
            mask_max_list.append(mask)
            mask_max, _indices = torch.max(torch.stack(mask_max_list, dim=0), dim=0)
            mask_overlap = mask_sum - mask_max

            # For the objects with low saliency score, m ~ U(0.0, 0.5)
            if score <= self.score_thresh:
                m_oa = np.float32(np.random.uniform(0.0, 0.5))
            else:
                m_oa = np.float32(np.random.uniform(0.0, 1.0))
            orig += (1.0 - m_oa) * feat * (mask - mask_overlap * 0.5)
            aug += m_oa * feat_aug * (mask - mask_overlap * 0.5)
            mask_sum = mask_max

        feat_oamix = orig + aug

        feat_oamix += (1.0 - m) * feat * (1.0 - mask_sum)
        feat_oamix += m * feat_aug * (1.0 - mask_sum)

        return feat_oamix

    def __repr__(self):
        return f"OAMix1D(severity={self.severity})"