# Clothes Segmentation with SegFormer on ATR

Pixel-accurate segmentation of the clothes a person is wearing, built for the
**CVCY002 Computer Vision Engineer assignment** (virtual fitting-room use case).

> **Submitted model: [`outputs_15epochs/`](outputs_15epochs/)** - SegFormer-B2, 15 epochs, trained on one laptop GPU.
> Held-out test split (885 images, original resolution): **mean IoU 0.750**, **clothes-vs-rest IoU 0.924**,
> pixel accuracy 0.967 (0.755 / 0.926 with `--tta`). Full analysis in [`reports/REPORT.pdf`](reports/REPORT.pdf).

* **Dataset:** [`mattmdjaga/human_parsing_dataset`](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) - the ATR human-parsing set, 17,706 photos, 18 classes (background, 6 body parts, 11 clothing/accessory classes).
* **Model:** [SegFormer-B2](https://arxiv.org/abs/2105.15203) initialised from `nvidia/segformer-b2-finetuned-ade-512-512`, fine-tuned end-to-end with an 18-class head.
* **Loss:** Cross-Entropy + 0.5 x soft Dice (handles the extreme class imbalance of small garments).
* **One command** runs the whole pipeline (download, split, train, evaluate, score the public reference, demo predictions) and stores metrics, training history and qualitative grids per run; the report in `reports/` is built from those files.

| Deliverable (assignment) | Where |
|---|---|
| Dataset preprocessing script | `src/clothseg/data.py`, `src/clothseg/prepare.py` |
| Model implementation | `src/clothseg/model.py`, `src/clothseg/losses.py` |
| Training and evaluation code | `src/clothseg/train.py`, `src/clothseg/evaluate.py`, `src/clothseg/metrics.py` |
| Inference pipeline (photo in -> clothes out) | `src/clothseg/predict.py` |
| Report (dataset, architecture, loss, metrics, limitations) | [`reports/REPORT.md`](reports/REPORT.md) / [`reports/REPORT.pdf`](reports/REPORT.pdf) |
| Code documentation (supplementary: module-by-module manual) | [`code_documentation/CODE_DOCUMENTATION.md`](code_documentation/CODE_DOCUMENTATION.md) / [`CODE_DOCUMENTATION.pdf`](code_documentation/CODE_DOCUMENTATION.pdf) |
| Single entry point | `run.py` |
| **Reported run** (metrics, history, config, eval figures) | [`outputs_15epochs/`](outputs_15epochs/) - see [`outputs_15epochs/README.md`](outputs_15epochs/README.md) |

---

## 1. Quick start

```bash
git clone <this-repo-url> clothes-segmentation-atr
cd clothes-segmentation-atr
pip install -r requirements.txt

python run.py check          # confirms torch / CUDA / libraries
python run.py all            # prepare -> train -> evaluate -> baseline -> predict demo
```

Everything is written to the `--output-dir` (default `outputs/`).
See section 2 for hardware requirements and runtimes.

### Smoke test (2-3 min on any GPU, also works on CPU)

```bash
python run.py all --limit 400 --epochs 1 --img-size 384 --skip-baseline
```

### Segment your own photo

```bash
python run.py predict --input path/to/photo.jpg            # one image
python run.py predict --input path/to/folder/               # every image in a folder
python run.py predict --from-test 8                         # first 8 test images incl. ground truth
python run.py predict --input photo.jpg --tta               # flip test-time augmentation: slower, slightly cleaner
```

Each image produces `<name>_mask.png` (18-class colour map), `<name>_overlay.jpg`,
`<name>_clothes.png` (transparent cut-out of the clothes only) and `<name>_labels.json`
(classes found and the percentage of the image they cover) in `outputs/predictions/`.

From Python:

```python
import sys; sys.path.insert(0, "src")
from PIL import Image
from clothseg.predict import ClothesSegmenter

seg = ClothesSegmenter("outputs/checkpoints/best")
mask = seg.predict(Image.open("photo.jpg"))   # (H, W) uint8 label map, ATR ids 0-17
```

---

## 2. Hardware and runtime

Everything in this repository was produced **locally on one laptop GPU**; no cloud account was used.

| | reported run |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, Windows 11 |
| settings | SegFormer-B2, 512x512, batch 4, bf16 mixed precision, `--num-workers 0` |
| training | 15 epochs, 3,984 steps / epoch, ~17.5 min / epoch, **4.4 h total** |
| evaluation | 885 test images at original resolution: ~1 min (53 img/s); with `--tta` ~4 min |
| disk | ~0.8 GB dataset cache (Hugging Face), ~110 MB per checkpoint |

Choosing settings for your GPU:

* **8 GB** (RTX 3060 / 4060 class): `--batch-size 4` as above; `--img-size 384` if you still run out of memory.
* **16 GB** (T4, RTX 4080 class): `--batch-size 4` is safe, `8` usually fits with mixed precision.
* **24 GB+** (L4, A10G, RTX 3090 / 4090, A100): the default `--batch-size 8`; roughly 45 min on an L4 / A10G
  and 25 min on an A100 for the default 10 epochs.
* **CPU only**: fine for the smoke test and for `predict` (about 0.5 s per photo), far too slow for full training.
* **Windows**: keep `--num-workers 0` (the default there); each DataLoader worker re-imports torch + CUDA and
  can exhaust memory.

If an epoch is interrupted, `python run.py train --resume` continues from `checkpoints/last`.

**No local GPU?** Any cloud notebook or studio with a terminal works: clone, `pip install -r requirements.txt`,
then run the same commands, prefixed with `nohup ... &` so the job survives a closed browser tab. A step-by-step
example for Lightning AI is in [`code_documentation/CODE_DOCUMENTATION.md`](code_documentation/CODE_DOCUMENTATION.md), section 9.

---

## 3. Commands

| command | what it does | main outputs |
|---|---|---|
| `python run.py check` | environment / GPU sanity check | - |
| `python run.py prepare` | download ATR, create the 90/5/5 split, dataset statistics, augmentation preview | `outputs/data/` |
| `python run.py train` | fine-tune SegFormer-B2, validate every epoch, keep best checkpoint | `outputs/checkpoints/{best,last}`, `outputs/history.csv`, `outputs/train_steps.csv` |
| `python run.py evaluate` | test-set metrics at original resolution, per-class table, qualitative grid | `outputs/eval/test_*` |
| `python run.py baseline` | same evaluation for the public `mattmdjaga/segformer_b2_clothes` model | `outputs/eval/baseline_*` |
| `python run.py predict` | segment clothes in your images / test images | `outputs/predictions/` |
| `python run.py all` | all of the above in order | everything |

Useful flags (all commands accept them; see `python run.py train -h`):

| flag | default | meaning |
|---|---|---|
| `--epochs` | 10 | training epochs |
| `--batch-size` | 8 | per-step batch size (use 4 on 8-16 GB GPUs) |
| `--img-size` | 512 | network input resolution |
| `--lr` | 6e-5 | AdamW learning rate (SegFormer recipe) |
| `--dice-weight` | 0.5 | weight of the Dice term (`0` = plain cross-entropy) |
| `--pretrained` | `nvidia/segformer-b2-finetuned-ade-512-512` | starting checkpoint (`nvidia/mit-b2` = ImageNet only, `nvidia/segformer-b5-finetuned-ade-640-640` = bigger model) |
| `--limit N` | off | use only N training images (quick experiments) |
| `--resume` | off | continue from `outputs/checkpoints/last` |
| `--no-amp` | off | disable mixed precision |
| `--tta` | off | horizontal-flip test-time augmentation for `evaluate` / `baseline` / `predict`: no retraining, +0.5 mIoU, ~4x slower inference; evaluation results go to `<output-dir>/eval/test_tta_*` |
| `--num-workers` | 4 (0 on Windows) | data-loading processes; keep 0 on Windows, each worker re-imports torch/CUDA there |
| `--output-dir` | `outputs` | where everything is written |

---

## 4. Project layout

```
clothes-segmentation-atr/
├── run.py                    # single entry point (all commands)
├── requirements.txt
├── README.md
├── src/clothseg/
│   ├── config.py             # dataset / label / colour constants + Config dataclass
│   ├── data.py               # HF dataset loading, deterministic split, augmentation, Dataset/DataLoader
│   ├── model.py              # SegFormer construction, save / load, device helpers
│   ├── losses.py             # Cross-Entropy + soft Dice
│   ├── metrics.py            # confusion-matrix metrics (mIoU, pixel acc, Dice, clothes-binary IoU)
│   ├── prepare.py            # download, split, statistics, augmentation preview
│   ├── train.py              # training loop (AMP, warm-up + poly LR, checkpoints, CSV logs)
│   ├── evaluate.py           # test evaluation at original resolution, baseline comparison
│   ├── predict.py            # inference pipeline: photo -> mask / overlay / clothes cut-out / json
│   └── visualize.py          # palette, overlays, grids
├── code_documentation/       # CODE_DOCUMENTATION.md + PDF: module-by-module manual (supplementary)
├── reports/REPORT.md         # technical report (+ figures, PDF)
├── outputs_15epochs/         # reported 15-epoch run: metrics, history, config (checkpoints / predictions git-ignored)
├── outputs_4epochs/          # first 4-epoch run, kept as a training-length ablation
└── outputs/                  # default --output-dir for new runs
```

---

## 5. Reproducing the results

The reported model was trained with:

```bash
python run.py all --epochs 15 --batch-size 4 --output-dir outputs_15epochs
```

The split is deterministic (`--seed 42`, stored in `<output-dir>/data/split.json`), the run configuration is
saved to `<output-dir>/run_config.json`, and all random generators are seeded. Re-running the command above on
the same GPU type reproduces the reported metrics up to GPU non-determinism (typically +-0.3 mIoU points).
Every run writes to its own `--output-dir`, so earlier runs are never overwritten: `outputs_4epochs/` holds the
first 4-epoch run (kept as a training-length ablation), `outputs_15epochs/` the reported 15-epoch model. The default
`outputs/` is free for fresh runs.

To evaluate an existing checkpoint without retraining:

```bash
python run.py evaluate --output-dir outputs_15epochs --checkpoint outputs_15epochs/checkpoints/best
```

---

## 6. Results

Test split (885 images, metrics at original resolution) after **15 epochs** on a single laptop GPU
(RTX 4060 8 GB, batch 4, 512x512, 4.4 h):

| metric | **ours (SegFormer-B2, 15 epochs)** | ours + flip TTA (`--tta`) | ours, earlier 4-epoch run | public `mattmdjaga/segformer_b2_clothes` * |
|---|---|---|---|---|
| mean IoU (18 classes) | **0.750** | 0.755 | 0.716 | 0.778 |
| mean IoU without background | 0.736 | 0.742 | 0.700 | 0.766 |
| pixel accuracy | **0.967** | 0.968 | 0.964 | 0.976 |
| mean class accuracy | 0.854 | 0.858 | 0.827 | 0.866 |
| mean Dice | 0.852 | 0.855 | 0.826 | 0.868 |
| clothes-vs-rest IoU (binary) | **0.924** | 0.926 | 0.916 | 0.932 |
| test loss (CE + 0.5 Dice) | 0.249 | 0.234 | 0.262 | 0.237 |
| inference speed (img/s, RTX 4060 laptop) | 52.9 | 12.5 | 52.9 | 53.6 |

\* scored with the same protocol on our test split; that model was trained on the whole ATR set (so it has
seen our test images) and for longer, which makes it an optimistic upper reference.

Validation mean IoU rose from 0.651 (epoch 1) to 0.752 (epoch 15) and was still improving slowly at the end;
validation loss bottomed out at epoch 9 while mIoU kept climbing, so there is no overfitting. Going from 4 to
15 epochs lifted every class except Skirt and Dress. Best classes: background, face, pants, upper-clothes,
hair (IoU > 0.82). Weakest: belt (0.45), scarf (0.60), shoes (0.65, mostly left/right swaps), sunglasses (0.66),
dress (0.68, confused with skirt / upper-clothes). Small accessories (belt, sunglasses, shoes) now beat the
public reference thanks to the Dice term. Flip test-time augmentation (`--tta`, same checkpoint, no retraining)
improves every class, most for the left / right ones (shoes, arms, legs +0.6 to +1.0 IoU points), at the cost of
a second forward pass.

![training curves](reports/figures/training_curves.png)
![prediction gallery](reports/figures/prediction_gallery.png)

See [`reports/REPORT.md`](reports/REPORT.md) / [`REPORT.pdf`](reports/REPORT.pdf) for the full analysis
(training curves, per-class IoU, confusion matrix, qualitative results, strengths / weaknesses /
capture-condition limits). Every number in it comes from `outputs_15epochs/eval/*.json` and `outputs_15epochs/history.csv`,
i.e. from the checkpoint in `outputs_15epochs/checkpoints/best`.

## License

MIT - see `LICENSE`. The ATR dataset and the SegFormer checkpoints keep their own licences
(research / non-commercial use for ATR).
