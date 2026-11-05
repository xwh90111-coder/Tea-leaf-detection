"""Train YOLOv12n on the tea-leaf-disease dataset.

Reproduces the training protocol described in the paper:
  * model      : YOLOv12n (yolo12.yaml, ImageNet/COCO pretrained weights)
  * epochs     : 400 (results.csv / Fig. 5 in the paper; table 2 lists 300)
  * batch      : 16 (paper table 2)
  * imgsz      : 640
  * patience   : 100
  * workers    : 8
  * optimizer  : auto (SGD -> AdamW auto-selection by Ultralytics)
  * AMP        : enabled
  * seed       : 0 / deterministic

Usage:
    python src/train.py                # full reproduction
    python src/train.py --epochs 1     # smoke test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce YOLOv12n tea disease training")
    p.add_argument("--model", default="yolo12n.pt", help="model checkpoint or yaml")
    p.add_argument("--data", default=str(ROOT / "data" / "tea.yaml"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="yolo12n_tea")
    p.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="ram", help="ram | disk | False")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    t0 = time.time()
    if args.resume:
        ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        model = YOLO(str(ckpt))
        results = model.train(resume=True, device=args.device)
        elapsed = time.time() - t0
        save_dir = Path(results.save_dir)
        meta = {"resumed_from": str(ckpt), "epoch": int(args.epochs) if hasattr(args, "epochs") else None,
                "elapsed_seconds": round(elapsed, 2), "save_dir": str(save_dir)}
        (save_dir / "resume_meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
        print(meta)
        return

    model = YOLO(args.model)
    params = sum(p.numel() for p in model.model.parameters())

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        amp=True,
        plots=True,
        val=True,
        cos_lr=False,
        close_mosaic=10,
        # Ultralytics' default augmentation pipeline (HSV / mosaic / fliplr ...)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cache=args.cache,
        resume=args.resume,
    )

    elapsed = time.time() - t0
    save_dir = Path(results.save_dir)
    meta = {
        "model": args.model,
        "parameters": params,
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 2),
        "save_dir": str(save_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": __import__("torch").__version__,
        "cuda": __import__("torch").version.cuda,
        "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None,
        "ultralytics": __import__("ultralytics").__version__,
    }
    (save_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
