from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


DATASET_NAME = "mattmdjaga/human_parsing_dataset"
DEFAULT_PRETRAINED = "nvidia/segformer-b2-finetuned-ade-512-512"
BASELINE_MODEL = "mattmdjaga/segformer_b2_clothes"


CLASS_NAMES = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


CLOTHING_CLASS_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]
BODY_CLASS_IDS = [2, 11, 12, 13, 14, 15]


FLIP_PAIRS = [(9, 10), (12, 13), (14, 15)]


PALETTE = [
    (0, 0, 0),
    (128, 0, 0),
    (255, 0, 0),
    (0, 85, 0),
    (170, 0, 51),
    (255, 85, 0),
    (0, 0, 85),
    (0, 119, 221),
    (85, 85, 0),
    (0, 85, 85),
    (85, 51, 0),
    (52, 86, 128),
    (0, 128, 0),
    (0, 0, 255),
    (51, 170, 221),
    (0, 255, 255),
    (85, 255, 170),
    (170, 255, 85),
]


DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 4


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:


    pretrained: str = DEFAULT_PRETRAINED
    img_size: int = 512


    epochs: int = 10
    batch_size: int = 8
    lr: float = 6e-5
    weight_decay: float = 0.01
    warmup_steps: int = 300
    grad_clip: float = 1.0
    dice_weight: float = 0.5
    amp: bool = True
    tta: bool = False


    val_fraction: float = 0.05
    test_fraction: float = 0.05
    limit: Optional[int] = None
    num_workers: int = DEFAULT_NUM_WORKERS
    seed: int = 42


    output_dir: str = "outputs"
    log_every: int = 20
    resume: bool = False
    device: str = "auto"


    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def ckpt_dir(self) -> Path:
        return self.out / "checkpoints"

    @property
    def best_dir(self) -> Path:
        return self.ckpt_dir / "best"

    @property
    def last_dir(self) -> Path:
        return self.ckpt_dir / "last"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)
