from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, NUM_CLASSES, PALETTE, Config
from .data import ATRDataset, denormalize, load_raw_dataset, make_splits, to_numpy_pair
from .visualize import colorize, save_grid


def prepare(cfg: Config, n_stats: int = 400) -> dict:
    t0 = time.time()
    out = cfg.out / "data"
    out.mkdir(parents=True, exist_ok=True)

    print(">> loading dataset (downloads ~800 MB on first run) ...")
    ds = load_raw_dataset()
    splits = make_splits(len(ds), cfg)
    print(f"   total={len(ds)}  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


    rng = np.random.default_rng(cfg.seed)
    sample_idx = rng.choice(splits["train"], size=min(n_stats, len(splits["train"])), replace=False)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_presence = np.zeros(NUM_CLASSES, dtype=np.int64)
    widths, heights = [], []
    for i in sample_idx:
        img, mask = to_numpy_pair(ds[int(i)])
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        pixel_counts += counts
        image_presence += counts > 0

    share = pixel_counts / pixel_counts.sum()
    stats = {
        "n_total": len(ds),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_stat_sample": int(len(sample_idx)),
        "image_width": {"min": int(min(widths)), "max": int(max(widths)), "mean": float(np.mean(widths))},
        "image_height": {"min": int(min(heights)), "max": int(max(heights)), "mean": float(np.mean(heights))},
        "class_pixel_share": {CLASS_NAMES[i]: float(share[i]) for i in range(NUM_CLASSES)},
        "class_image_presence": {CLASS_NAMES[i]: float(image_presence[i] / len(sample_idx)) for i in range(NUM_CLASSES)},
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))


    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [np.array(c) / 255.0 for c in PALETTE]
    ax.bar(range(NUM_CLASSES), share * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("pixel share (%)  [log]")
    ax.set_title(f"ATR class pixel share (sample of {len(sample_idx)} training images)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "class_frequency.png", dpi=120)
    plt.close(fig)


    preview_idx = splits["train"][:3]
    plain = ATRDataset(ds, preview_idx, cfg.img_size, train=False)
    aug = ATRDataset(ds, preview_idx, cfg.img_size, train=True)
    rows = []
    for k in range(len(preview_idx)):
        base = plain[k]
        row = [denormalize(base["pixel_values"]), colorize(base["labels"].numpy().astype(np.uint8))]
        for _ in range(3):
            a = aug[k]
            row.append(denormalize(a["pixel_values"]))
            row.append(colorize(a["labels"].numpy().astype(np.uint8)))
        rows.append(row)
    titles = ["image", "mask"] + [t for _ in range(3) for t in ("aug image", "aug mask")]
    save_grid(rows, titles, out / "augmentation_preview.png", legend=True, cell=2.2)

    print(f"   image size: w {stats['image_width']}  h {stats['image_height']}")
    print(f"   rarest classes by pixel share: "
          + ", ".join(f"{CLASS_NAMES[i]} {share[i]*100:.2f}%" for i in np.argsort(share)[:5]))
    print(f">> prepare done in {time.time()-t0:.1f}s -> {out}")
    return stats
