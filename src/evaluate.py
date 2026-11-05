"""Evaluate the trained YOLOv12n on the held-out test / validation splits.

Outputs (under reports/ and runs/detect/<name>/):
  * metrics_<split>.json   overall + per-class metrics
  * per_class_<split>.csv
  * PR / F1 / confusion-matrix figures (produced by Ultralytics)

Usage:
    python src/evaluate.py --weights runs/detect/yolo12n_tea_400/weights/best.pt --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/detect/yolo12n_tea_400/weights/best.pt"))
    p.add_argument("--data", default=str(ROOT / "data/tea.yaml"))
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default=None)
    return p.parse_args()


def run_val(weights: str, data: str, split: str, imgsz: int, batch: int, device: str, name: str, conf: float):
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=0.7,
        plots=True,
        project=str(ROOT / "runs" / "val"),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    names = metrics.names
    per_class = []
    for i, cid in enumerate(box.ap_class_index):
        name_cn = names.get(int(cid), str(cid))
        per_class.append(
            {
                "class_id": int(cid),
                "class_name": name_cn,
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "f1": float(box.f1[i]),
                "ap50": float(box.ap50[i]),
                "ap50_95": float(box.ap[i]),
                "instances": int(np.asarray(metrics.confusion_matrix.matrix).sum(1)[int(cid)])
                if hasattr(metrics, "confusion_matrix") and hasattr(metrics.confusion_matrix, "matrix")
                else None,
            }
        )
    overall = {
        "split": split,
        "conf": conf,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": float(box.f1.mean()),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "classes": len(per_class),
        "speed_ms": {k: round(float(v), 3) for k, v in metrics.speed.items()},
    }
    save_dir = Path(metrics.save_dir)
    return overall, per_class, save_dir


def main():
    args = parse_args()
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.split}_eval"

    # mAP mode (low confidence, full PR curve)
    overall, per_class, save_dir = run_val(
        args.weights, args.data, args.split, args.imgsz, args.batch, args.device, name, conf=0.001
    )
    # operating point (paper table reports Precision/Recall at a fixed threshold)
    overall_op, per_class_op, _ = run_val(
        args.weights, args.data, args.split, args.imgsz, args.batch, args.device, name + "_conf025", conf=0.25
    )

    payload = {
        "map_mode": overall,
        "operating_point_conf025": overall_op,
        "per_class": per_class,
        "per_class_conf025": per_class_op,
    }
    (REPORTS / f"metrics_{args.split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (REPORTS / f"per_class_{args.split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1", "ap50", "ap50_95"])
        w.writeheader()
        for row in per_class:
            w.writerow({k: row[k] for k in w.fieldnames})

    for src in save_dir.glob("*.png"):
        if "_curve" in src.name or "confusion_matrix" in src.name:
            shutil.copy2(src, FIGS / f"{args.split}_{src.name}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
