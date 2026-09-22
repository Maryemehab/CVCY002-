from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IGNORE_INDEX, NUM_CLASSES


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX, eps: float = 1.0) -> torch.Tensor:
    num_classes = logits.shape[1]
    valid = (target != ignore_index)
    safe_target = target.masked_fill(~valid, 0)

    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)

    valid = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * valid
    one_hot = one_hot * valid

    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):

    def __init__(self, dice_weight: float = 0.5, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        if self.dice_weight > 0:
            dice = soft_dice_loss(logits, target, self.ignore_index)
        else:
            dice = torch.zeros((), device=logits.device)
        total = ce + self.dice_weight * dice
        return total, {"ce": ce.detach().item(), "dice": dice.detach().item()}


def upsample_logits(logits: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
