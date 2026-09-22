# Reported run - SegFormer-B2, 15 epochs (the submitted model)

This folder holds the run behind every number in `reports/REPORT.pdf` and the main README.

| file | content |
|---|---|
| `run_config.json` | exact configuration (`python run.py all --epochs 15 --batch-size 4 --output-dir outputs_15epochs`) |
| `train_summary.json` | best validation mIoU, epochs, parameter count, training time, GPU used |
| `history.csv` / `history.json` | per-epoch train / validation loss and metrics |
| `train_steps.csv` | training loss every 20 steps (for the loss curve) |
| `eval/test_metrics.json` | test-split metrics at original resolution incl. per-class IoU and confusion matrix |
| `eval/test_tta_metrics.json` | same, with horizontal-flip test-time augmentation (`--tta`) |
| `eval/baseline_metrics.json` | the public `mattmdjaga/segformer_b2_clothes` model scored with the same protocol |
| `eval/*_per_class.csv`, `eval/*_qualitative.png` | per-class tables and qualitative grids |
| `data/split.json`, `data/dataset_stats.json` | the deterministic 90 / 5 / 5 split and dataset statistics |
| `checkpoints/best/` | the trained model (Hugging Face format, ~110 MB) - **git-ignored**, produced by training |

Headline: test mean IoU **0.750** (0.755 with `--tta`), clothes-vs-rest IoU **0.924**, pixel accuracy 0.967.
