from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .config import BASELINE_MODEL, CLASS_NAMES, Config
from .data import ATRDataset, denormalize, load_raw_dataset, make_loader, make_splits
from .losses import SegmentationLoss, upsample_logits
from .metrics import SegMetrics, format_summary, per_class_table
from .model import autocast_dtype, count_parameters, forward_logits, load_model, pick_device
from .visualize import colorize, save_grid


@torch.no_grad()
def run_validation(model, loader, loss_fn, device, amp_dtype) -> dict:
    model.eval()
    metrics = SegMetrics(device=device)
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(pixel_values=x).logits
        logits = upsample_logits(logits.float(), y.shape[-2:])
        loss, _ = loss_fn(logits, y)
        total_loss += loss.item() * x.shape[0]
        n += x.shape[0]
        metrics.update(logits.argmax(1), y)
    model.train()
    return {"loss": total_loss / max(1, n), "metrics": metrics.compute()}


@torch.no_grad()
def evaluate(cfg: Config, checkpoint: Optional[Path] = None, hub_model: Optional[str] = None,
             tag: str = "test", split: str = "test", n_qualitative: int = 6) -> dict:
    device = pick_device(cfg.device)
    amp_dtype = autocast_dtype(device, cfg.amp)
    source = hub_model or str(checkpoint or cfg.best_dir)
    if cfg.tta and not tag.endswith("_tta"):
        tag = f"{tag}_tta"
    print(f">> evaluating {source} on '{split}' split at original resolution{' with flip TTA' if cfg.tta else ''}")

    model = load_model(source).to(device).eval()
    ds = load_raw_dataset()
    splits = make_splits(len(ds), cfg)
    eval_ds = ATRDataset(ds, splits[split], cfg.img_size, train=False, return_original=True)
    loader = make_loader(eval_ds, cfg, shuffle=False)
    loss_fn = SegmentationLoss(dice_weight=cfg.dice_weight)

    metrics_full = SegMetrics(device=device)
    metrics_net = SegMetrics(device=device)
    total_loss, n_img, infer_time = 0.0, 0, 0.0
    qual_rows = []

    for batch in tqdm(loader, desc=f"eval[{tag}]", ncols=100):
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits = forward_logits(model, x, amp_dtype, tta=cfg.tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time += time.time() - t0

        net_logits = upsample_logits(logits, y.shape[-2:])
        loss, _ = loss_fn(net_logits, y)
        total_loss += loss.item() * x.shape[0]
        n_img += x.shape[0]
        net_pred = net_logits.argmax(1)
        metrics_net.update(net_pred, y)

        for i, orig in enumerate(batch["orig_mask"]):
            up = upsample_logits(logits[i : i + 1], orig.shape[-2:]).argmax(1)[0]
            metrics_full.update(up, orig.to(device))

        if len(qual_rows) < n_qualitative:
            for i in range(min(x.shape[0], n_qualitative - len(qual_rows))):
                qual_rows.append([
                    denormalize(x[i]),
                    colorize(y[i].cpu().numpy().astype(np.uint8)),
                    colorize(net_pred[i].cpu().numpy().astype(np.uint8)),
                ])

    result = metrics_full.compute()
    result.update({
        "source": source,
        "split": split,
        "n_images": n_img,
        "loss": total_loss / max(1, n_img),
        "net_resolution_mean_iou": metrics_net.compute()["mean_iou"],
        "img_size": cfg.img_size,
        "params": count_parameters(model),
        "images_per_second": n_img / max(infer_time, 1e-6),
        "device": str(device),
        "tta": bool(cfg.tta),
    })

    out = cfg.out / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2))
    with (out / f"{tag}_per_class.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "iou", "acc", "dice", "support_px"])
        for i, name in enumerate(CLASS_NAMES):
            r = result["per_class"][name]
            w.writerow([i, name, f"{r['iou']:.4f}", f"{r['acc']:.4f}", f"{r['dice']:.4f}", r["support_px"]])
    if qual_rows:
        save_grid(qual_rows, ["image", "ground truth", "prediction"], out / f"{tag}_qualitative.png")

    print(f"   {format_summary(result)}")
    print(f"   loss={result['loss']:.4f}  speed={result['images_per_second']:.1f} img/s  n={n_img}")
    print(per_class_table(result))
    print(f">> saved -> {out / (tag + '_metrics.json')}")
    return result


def evaluate_baseline(cfg: Config) -> dict:
    return evaluate(cfg, hub_model=BASELINE_MODEL, tag="baseline")
