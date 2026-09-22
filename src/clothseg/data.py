from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from datasets import load_dataset

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    DATASET_NAME,
    FLIP_PAIRS,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Config,
)


def load_raw_dataset():
    return load_dataset(DATASET_NAME, split="train")


def make_splits(n_total: int, cfg: Config, split_file: Optional[Path] = None) -> Dict[str, List[int]]:
    split_file = split_file or cfg.out / "data" / "split.json"
    if split_file.exists():
        splits = json.loads(split_file.read_text())
        if sum(len(v) for v in splits.values()) == n_total:
            pass
        else:
            splits = None
    else:
        splits = None

    if splits is None:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n_total).tolist()
        n_test = int(round(n_total * cfg.test_fraction))
        n_val = int(round(n_total * cfg.val_fraction))
        splits = {
            "test": sorted(perm[:n_test]),
            "val": sorted(perm[n_test : n_test + n_val]),
            "train": sorted(perm[n_test + n_val :]),
        }
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(splits))

    if cfg.limit:
        small = max(16, cfg.limit // 8)
        splits = {
            "train": splits["train"][: cfg.limit],
            "val": splits["val"][:small],
            "test": splits["test"][:small],
        }
    return splits


def swap_left_right(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for a, b in FLIP_PAIRS:
        out[mask == a] = b
        out[mask == b] = a
    return out


def build_transforms(img_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Affine(
                    scale=(0.75, 1.25),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-12, 12),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=IGNORE_INDEX,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def to_numpy_pair(example) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(example["image"].convert("RGB"), dtype=np.uint8)
    mask = np.asarray(example["mask"])
    if mask.ndim == 3:
        mask = mask[..., 0]
    return image, mask.astype(np.uint8)


class ATRDataset(Dataset):

    def __init__(self, hf_ds, indices: Sequence[int], img_size: int, train: bool, return_original: bool = False):
        self.ds = hf_ds
        self.indices = list(indices)
        self.train = train
        self.return_original = return_original
        self.tf = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        image, mask = to_numpy_pair(self.ds[int(self.indices[i])])

        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(swap_left_right(mask[:, ::-1]))

        out = self.tf(image=image, mask=mask)
        item = {
            "pixel_values": torch.from_numpy(out["image"]).permute(2, 0, 1).float(),
            "labels": torch.from_numpy(out["mask"].astype(np.int64)),
            "index": int(self.indices[i]),
        }
        if self.return_original:
            item["orig_mask"] = torch.from_numpy(mask.astype(np.int64))
        return item


def collate_keep_original(batch: List[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "index": [b["index"] for b in batch],
    }
    if "orig_mask" in batch[0]:
        out["orig_mask"] = [b["orig_mask"] for b in batch]
    return out


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate_keep_original,
    )


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
