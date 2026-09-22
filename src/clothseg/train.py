from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import Config
from .data import ATRDataset, load_raw_dataset, make_loader, make_splits
from .evaluate import run_validation
from .losses import SegmentationLoss, upsample_logits
from .metrics import format_summary
from .model import autocast_dtype, build_model, count_parameters, load_model, pick_device, save_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_schedule(total_steps: int, warmup: int, power: float = 1.0):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, (1 - progress)) ** power
    return fn


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    amp_dtype = autocast_dtype(device, cfg.amp)
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.out / "run_config.json")
    print(f">> device={device}  amp={amp_dtype}")


    ds = load_raw_dataset()
    splits = make_splits(len(ds), cfg)
    train_ds = ATRDataset(ds, splits["train"], cfg.img_size, train=True)
    val_ds = ATRDataset(ds, splits["val"], cfg.img_size, train=False)
    train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    print(f"   train={len(train_ds)} val={len(val_ds)}  steps/epoch={len(train_loader)}")


    resume_from = cfg.last_dir if cfg.resume and (cfg.last_dir / "config.json").exists() else None
    model = load_model(resume_from) if resume_from else build_model(cfg.pretrained)
    model.to(device)
    n_params = count_parameters(model)
    print(f"   model={cfg.pretrained}  params={n_params['total']/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, poly_schedule(total_steps, cfg.warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    loss_fn = SegmentationLoss(dice_weight=cfg.dice_weight)

    history, best_miou, start_epoch, global_step = [], -1.0, 0, 0
    state_path = cfg.ckpt_dir / "training_state.pt"
    if resume_from and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        history, best_miou = st["history"], st["best_miou"]
        start_epoch, global_step = st["epoch"] + 1, st["global_step"]
        print(f"   resumed from epoch {st['epoch']} (best mIoU {best_miou:.4f})")
    else:
        for p in (cfg.out / "history.csv", cfg.out / "train_steps.csv"):
            if p.exists():
                p.unlink()


    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t_epoch = time.time()
        run = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", ncols=110)
        for batch in pbar:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(pixel_values=x).logits
            logits = upsample_logits(logits.float(), y.shape[-2:])
            loss, parts = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            run["loss"] += loss.item(); run["ce"] += parts["ce"]; run["dice"] += parts["dice"]; run["n"] += 1
            if global_step % cfg.log_every == 0:
                _append_csv(cfg.out / "train_steps.csv", {
                    "step": global_step, "epoch": epoch + 1, "loss": round(loss.item(), 5),
                    "ce": round(parts["ce"], 5), "dice": round(parts["dice"], 5),
                    "lr": scheduler.get_last_lr()[0],
                })
                pbar.set_postfix(loss=f"{run['loss']/run['n']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")


        val = run_validation(model, val_loader, loss_fn, device, amp_dtype)
        row = {
            "epoch": epoch + 1,
            "train_loss": run["loss"] / max(1, run["n"]),
            "train_ce": run["ce"] / max(1, run["n"]),
            "train_dice": run["dice"] / max(1, run["n"]),
            "val_loss": val["loss"],
            "val_mean_iou": val["metrics"]["mean_iou"],
            "val_pixel_acc": val["metrics"]["pixel_acc"],
            "val_mean_acc": val["metrics"]["mean_acc"],
            "val_mean_dice": val["metrics"]["mean_dice"],
            "val_clothes_iou": val["metrics"]["clothes_binary_iou"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": round(time.time() - t_epoch, 1),
        }
        history.append(row)
        _append_csv(cfg.out / "history.csv", row)
        (cfg.out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"   epoch {epoch+1}: train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
              f"| {format_summary(val['metrics'])} | {row['epoch_time_s']}s")


        save_model(model, cfg.last_dir, cfg.img_size)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "best_miou": best_miou, "epoch": epoch, "global_step": global_step}, state_path)
        if row["val_mean_iou"] > best_miou:
            best_miou = row["val_mean_iou"]
            save_model(model, cfg.best_dir, cfg.img_size)
            (cfg.best_dir / "val_metrics.json").write_text(json.dumps({**val["metrics"], "epoch": epoch + 1, "val_loss": val["loss"]}, indent=2))
            print(f"   ** new best val mIoU {best_miou:.4f} -> {cfg.best_dir}")

    summary = {"best_val_mean_iou": best_miou, "epochs_run": len(history), "params": n_params,
               "total_train_time_s": round(sum(h["epoch_time_s"] for h in history), 1),
               "device": str(device), "amp_dtype": str(amp_dtype),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device.type == "cuda" else None}
    (cfg.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f">> training finished. best val mIoU={best_miou:.4f}")
    return summary
