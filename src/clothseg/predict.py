from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .config import CLASS_NAMES, CLOTHING_CLASS_IDS, IMAGENET_MEAN, IMAGENET_STD, Config
from .losses import upsample_logits
from .model import autocast_dtype, forward_logits, load_model, pick_device
from .visualize import clothes_cutout, colorize, overlay

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClothesSegmenter:
    def __init__(self, checkpoint: Union[str, Path], img_size: int = 512, device: str = "auto", amp: bool = True, tta: bool = False):
        self.device = pick_device(device)
        self.amp_dtype = autocast_dtype(self.device, amp)
        self.img_size = img_size
        self.tta = tta
        self.model = load_model(checkpoint).to(self.device).eval()
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self._std = np.array(IMAGENET_STD, dtype=np.float32)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        x = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - self._mean) / self._std
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        h, w = image.shape[:2]
        x = self._preprocess(image)
        logits = forward_logits(self.model, x, self.amp_dtype, tta=self.tta)
        pred = upsample_logits(logits, (h, w)).argmax(1)[0]
        return pred.cpu().numpy().astype(np.uint8)

    def segment_to_files(self, image: np.ndarray, out_dir: Path, stem: str) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = self.predict(image)
        paths = {
            "mask": out_dir / f"{stem}_mask.png",
            "overlay": out_dir / f"{stem}_overlay.jpg",
            "clothes": out_dir / f"{stem}_clothes.png",
            "labels": out_dir / f"{stem}_labels.json",
        }
        Image.fromarray(colorize(mask)).save(paths["mask"])
        Image.fromarray(overlay(image, mask)).save(paths["overlay"], quality=92)
        Image.fromarray(clothes_cutout(image, mask), mode="RGBA").save(paths["clothes"])

        total = mask.size
        present = {CLASS_NAMES[c]: round(float((mask == c).mean() * 100), 2) for c in np.unique(mask) if c != 0}
        summary = {
            "image": stem,
            "size": [int(mask.shape[1]), int(mask.shape[0])],
            "classes_present": present,
            "clothes_pixel_percent": round(float(np.isin(mask, CLOTHING_CLASS_IDS).sum() / total * 100), 2),
        }
        paths["labels"].write_text(json.dumps(summary, indent=2))
        return {"mask_array": mask, "summary": summary, "paths": {k: str(v) for k, v in paths.items()}}


def _collect_inputs(inp: Path) -> List[Path]:
    if inp.is_dir():
        return sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXT)
    return [inp]


def run_predict(cfg: Config, inputs: Optional[str], from_test: int, checkpoint: Optional[str], out_dir: Optional[str]) -> List[dict]:
    ckpt = checkpoint or str(cfg.best_dir)
    out = Path(out_dir) if out_dir else cfg.out / "predictions"
    seg = ClothesSegmenter(ckpt, img_size=cfg.img_size, device=cfg.device, amp=cfg.amp, tta=cfg.tta)
    print(f">> model={ckpt}  device={seg.device}  tta={cfg.tta}  -> {out}")
    results = []

    if inputs:
        for p in _collect_inputs(Path(inputs)):
            img = np.asarray(Image.open(p).convert("RGB"))
            r = seg.segment_to_files(img, out, p.stem)
            results.append(r["summary"])
            print(f"   {p.name}: clothes={r['summary']['clothes_pixel_percent']}%  classes={list(r['summary']['classes_present'])}")

    if from_test > 0:
        from .data import load_raw_dataset, make_splits, to_numpy_pair
        ds = load_raw_dataset()
        splits = make_splits(len(ds), cfg)
        for k, idx in enumerate(splits["test"][:from_test]):
            img, gt = to_numpy_pair(ds[int(idx)])
            stem = f"test_{k:02d}_idx{idx}"
            r = seg.segment_to_files(img, out, stem)
            Image.fromarray(img).save(out / f"{stem}_input.jpg", quality=92)
            Image.fromarray(colorize(gt)).save(out / f"{stem}_gt.png")
            results.append(r["summary"])
            print(f"   {stem}: clothes={r['summary']['clothes_pixel_percent']}%  classes={list(r['summary']['classes_present'])}")

    (out / "predictions_summary.json").write_text(json.dumps(results, indent=2))
    return results
