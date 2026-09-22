from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import clothseg
from clothseg.config import Config


def add_common(p: argparse.ArgumentParser) -> None:
    d = Config()
    p.add_argument("--pretrained", default=d.pretrained, help="HF checkpoint to start from")
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--dice-weight", type=float, default=d.dice_weight)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--tta", action="store_true",
                   help="flip test-time augmentation for evaluate / baseline / predict (no retraining, slower inference, ~+0.5 mIoU)")
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--test-fraction", type=float, default=d.test_fraction)
    p.add_argument("--limit", type=int, default=None, help="use only N training images (smoke test)")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/last")
    p.add_argument("--device", default=d.device, help="auto | cuda | cpu")


def cfg_from_args(a: argparse.Namespace) -> Config:
    return Config(
        pretrained=a.pretrained, img_size=a.img_size, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, warmup_steps=a.warmup_steps, grad_clip=a.grad_clip, dice_weight=a.dice_weight,
        amp=not a.no_amp, tta=a.tta, val_fraction=a.val_fraction, test_fraction=a.test_fraction, limit=a.limit,
        num_workers=a.num_workers, seed=a.seed, output_dir=a.output_dir, log_every=a.log_every,
        resume=a.resume, device=a.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clothes segmentation on ATR with SegFormer", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_ in [("check", "environment sanity check"), ("prepare", "download + split + stats"),
                        ("train", "fine-tune SegFormer"), ("evaluate", "test-set evaluation"),
                        ("baseline", "evaluate public reference model"), ("predict", "segment clothes in images"),
                        ("all", "run the full pipeline")]:
        p = sub.add_parser(name, help=help_)
        add_common(p)
        if name in ("evaluate", "predict"):
            p.add_argument("--checkpoint", default=None, help="model folder (default: outputs/checkpoints/best)")
        if name == "predict":
            p.add_argument("--input", default=None, help="image file or folder")
            p.add_argument("--from-test", type=int, default=0, help="also segment the first N test images")
            p.add_argument("--out", default=None, help="output folder (default: outputs/predictions)")
        if name == "all":
            p.add_argument("--skip-baseline", action="store_true", help="do not score the public reference model")
            p.add_argument("--demo-images", type=int, default=6, help="test images to run the predict demo on")
    return parser


def cmd_check(cfg: Config) -> None:
    import torch, transformers, datasets, albumentations
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  gpu={torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else ""))
    print(f"transformers {transformers.__version__}   datasets {datasets.__version__}   albumentations {albumentations.__version__}")
    print(f"output dir   {cfg.out.resolve()}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    cmd = args.command

    if cmd == "check":
        cmd_check(cfg); return

    if cmd in ("prepare", "all"):
        from clothseg.prepare import prepare
        prepare(cfg)

    if cmd in ("train", "all"):
        from clothseg.train import train
        train(cfg)

    if cmd in ("evaluate", "all"):
        from clothseg.evaluate import evaluate
        ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
        evaluate(cfg, checkpoint=ckpt, tag="test")

    if cmd == "baseline" or (cmd == "all" and not args.skip_baseline):
        from clothseg.evaluate import evaluate_baseline
        evaluate_baseline(cfg)

    if cmd == "predict" or (cmd == "all" and args.demo_images > 0):
        from clothseg.predict import run_predict
        if cmd == "predict":
            if not args.input and args.from_test == 0:
                sys.exit("predict: give --input <file|folder> and/or --from-test N")
            run_predict(cfg, args.input, args.from_test, args.checkpoint, args.out)
        else:
            run_predict(cfg, None, args.demo_images, None, None)



if __name__ == "__main__":
    main()
