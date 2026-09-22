from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .config import CLASS_NAMES, FLIP_PAIRS, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

ID2LABEL = {i: n for i, n in enumerate(CLASS_NAMES)}
LABEL2ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def build_model(pretrained: str) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model


def load_model(path_or_name: Union[str, Path]) -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation.from_pretrained(str(path_or_name))


def save_model(model: SegformerForSemanticSegmentation, folder: Path, img_size: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    SegformerImageProcessor(
        do_resize=True,
        size={"height": img_size, "width": img_size},
        do_reduce_labels=False,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
    ).save_pretrained(folder)


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(device: torch.device, amp: bool):
    if not amp or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def swap_flip_channels(logits: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.shape[1], device=logits.device)
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    return logits[:, idx]


@torch.no_grad()
def forward_logits(model, x: torch.Tensor, amp_dtype, tta: bool = False) -> torch.Tensor:
    device = x.device
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(pixel_values=x).logits
    logits = logits.float()
    if not tta:
        return logits
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits_f = model(pixel_values=torch.flip(x, dims=[3])).logits
    logits_f = swap_flip_channels(torch.flip(logits_f.float(), dims=[3]))
    probs = 0.5 * (logits.softmax(1) + logits_f.softmax(1))
    return probs.clamp_min(1e-8).log()
