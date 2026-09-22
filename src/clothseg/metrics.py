from __future__ import annotations

import torch

from .config import CLASS_NAMES, CLOTHING_CLASS_IDS, IGNORE_INDEX, NUM_CLASSES


def _nanmean(x: torch.Tensor) -> float:
    m = ~torch.isnan(x)
    return float(x[m].mean()) if m.any() else float("nan")


class SegMetrics:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX, device="cpu"):
        self.C = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.to(self.cm.device).flatten()
        target = target.to(self.cm.device).flatten()
        valid = target != self.ignore_index
        idx = target[valid] * self.C + pred[valid]
        self.cm += torch.bincount(idx, minlength=self.C * self.C).view(self.C, self.C)

    def compute(self) -> dict:
        cm = self.cm.double()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        iou = tp / (tp + fp + fn)
        acc = tp / (tp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)

        cloth = torch.zeros(self.C, dtype=torch.bool, device=cm.device)
        cloth[CLOTHING_CLASS_IDS] = True
        tp_c = cm[cloth][:, cloth].sum()
        fn_c = cm[cloth][:, ~cloth].sum()
        fp_c = cm[~cloth][:, cloth].sum()

        result = {
            "pixel_acc": float(tp.sum() / cm.sum().clamp(min=1)),
            "mean_acc": _nanmean(acc),
            "mean_iou": _nanmean(iou),
            "fg_mean_iou": _nanmean(iou[1:]),
            "mean_dice": _nanmean(dice),
            "clothes_binary_iou": float(tp_c / (tp_c + fp_c + fn_c).clamp(min=1)),
            "clothes_binary_dice": float(2 * tp_c / (2 * tp_c + fp_c + fn_c).clamp(min=1)),
            "per_class": {
                CLASS_NAMES[i]: {
                    "iou": float(iou[i]),
                    "acc": float(acc[i]),
                    "dice": float(dice[i]),
                    "support_px": int(cm[i].sum()),
                }
                for i in range(self.C)
            },
            "confusion_matrix": self.cm.cpu().tolist(),
        }
        return result


def format_summary(m: dict) -> str:
    keys = ["mean_iou", "fg_mean_iou", "pixel_acc", "mean_acc", "mean_dice", "clothes_binary_iou", "clothes_binary_dice"]
    return " | ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


def per_class_table(m: dict) -> str:
    rows = ["| id | class | IoU | Acc | Dice | pixels |", "|---|---|---|---|---|---|"]
    for i, name in enumerate(CLASS_NAMES):
        r = m["per_class"][name]
        rows.append(f"| {i} | {name} | {r['iou']:.3f} | {r['acc']:.3f} | {r['dice']:.3f} | {r['support_px']:,} |")
    return "\n".join(rows)
