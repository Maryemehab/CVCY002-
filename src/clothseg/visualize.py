from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, CLOTHING_CLASS_IDS, IGNORE_INDEX, PALETTE

_PALETTE = np.array(PALETTE, dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != IGNORE_INDEX
    out[valid] = _PALETTE[mask[valid]]
    out[~valid] = 160
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = colorize(mask)
    fg = mask != 0
    blended = image.astype(np.float32).copy()
    blended[fg] = (1 - alpha) * blended[fg] + alpha * color[fg]
    return blended.astype(np.uint8)


def clothes_cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    keep = np.isin(mask, CLOTHING_CLASS_IDS)
    rgba = np.concatenate([image, np.zeros((*mask.shape, 1), dtype=np.uint8)], axis=-1)
    rgba[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return rgba


def legend_handles(class_ids: Optional[Sequence[int]] = None):
    from matplotlib.patches import Patch

    ids = class_ids if class_ids is not None else range(len(CLASS_NAMES))
    return [Patch(facecolor=np.array(PALETTE[i]) / 255.0, label=f"{i} {CLASS_NAMES[i]}") for i in ids]


def save_grid(rows: List[List[np.ndarray]], titles: List[str], path: Path, legend: bool = True, cell: float = 3.0) -> None:
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows * 1.15), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r][c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
    if legend:
        fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.subplots_adjust(bottom=0.12)
    fig.tight_layout(rect=(0, 0.1 if legend else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
