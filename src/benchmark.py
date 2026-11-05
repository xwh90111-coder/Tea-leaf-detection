"""Benchmark parameters, GFLOPs, file size and inference speed.

The paper reports 2,557,313 parameters, 6.3 GFLOPs, 5.3 MB, 0.655 ms
preprocess + 3.764 ms inference + 0.863 ms postprocess = 189.30 FPS.

Usage:
    python src/benchmark.py --weights runs/detect/yolo12n_tea_400/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/detect/yolo12n_tea_400/weights/best.pt"))
    p.add_argument("--image", default=str(ROOT / "data/samples/sample_severe.jpg"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--runs-wall", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    weights = Path(args.weights)
    model = YOLO(str(weights))
    # Fuse Conv+BN (the same graph used by deployment); unfused checkpoints
    # report extra parameters and are slower.
    model.fuse()
    n_params = sum(p.numel() for p in model.model.parameters())
    flops = float(get_flops(model.model, args.imgsz))
    file_mb = weights.stat().st_size / 1024 / 1024

    image = np.array(Image.open(args.image).convert("RGB"))[:, :, ::-1]
    for _ in range(args.warmup):
        model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)

    speeds = {"preprocess": [], "inference": [], "postprocess": []}
    walls = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        r = model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        walls.append((time.perf_counter() - t0) * 1000)
        for k in speeds:
            speeds[k].append(float(r.speed.get(k, 0.0)))

    pre = statistics.mean(speeds["preprocess"])
    inf = statistics.mean(speeds["inference"])
    post = statistics.mean(speeds["postprocess"])
    end2end = pre + inf + post
    wall_mean = statistics.mean(walls[5:]) if len(walls) > 5 else statistics.mean(walls)

    batching = {}
    if args.device != "cpu":
        for bs in (4, 16):
            x = torch.randn(bs, 3, args.imgsz, args.imgsz, device="cuda")
            with torch.no_grad():
                for _ in range(10):
                    model.model(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(40):
                    model.model(x)
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 40 * 1000
            batching[f"batch{bs}_ms"] = round(ms, 3)
            batching[f"batch{bs}_fps_per_image"] = round(bs * 1000 / ms, 2)

    result = {
        "weights": str(weights),
        "parameters": int(n_params),
        "gflops": round(flops, 3),
        "model_file_mb": round(file_mb, 3),
        "imgsz": args.imgsz,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else "cpu",
        "runs": args.runs,
        "preprocess_ms": round(pre, 4),
        "inference_ms": round(inf, 4),
        "postprocess_ms": round(post, 4),
        "end2end_ms": round(end2end, 4),
        "fps_inference_only": round(1000 / inf, 2) if inf else None,
        "fps_end2end": round(1000 / end2end, 2) if end2end else None,
        "wall_ms_mean": round(wall_mean, 3),
        "fps_wall": round(1000 / wall_mean, 2) if wall_mean else None,
        "p95_inference_ms": round(float(np.percentile(speeds["inference"], 95)), 4),
        "p95_end2end_ms": round(float(np.percentile([a + b + c for a, b, c in zip(speeds["preprocess"], speeds["inference"], speeds["postprocess"])], 95)), 4),
        "batching": batching,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
