from typing import List

import torch
import torch.nn.functional as F


def masked_l2_loss(pred, target, weight_known, weight_missing):
    per_pixel_l2 = F.mse_loss(pred, target, reduction='none')
    return (per_pixel_l2).mean()


def masked_l1_loss(pred, target, weight_known, weight_missing):
    per_pixel_l1 = F.l1_loss(pred, target, reduction='none')
    return (per_pixel_l1).mean() * weight_known


def feature_matching_loss(fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_val = ((fake_feat - target_feat).pow(2)).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res
