"""Plot training curves from an Ultralytics results.csv.

Usage:
    python src/plot_training.py --csv runs/detect/yolo12n_tea_400/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "runs/detect/yolo12n_tea_400/results.csv"))
    p.add_argument("--out", default=str(FIGS / "training_curves.png"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=170)
    ax = axes[0]
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", color="#1f6f43", lw=2)
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#c8a24a", lw=2)
    best = df.loc[df["metrics/mAP50(B)"].idxmax()]
    ax.scatter([best["epoch"]], [best["metrics/mAP50(B)"]], color="#c0392b", zorder=5)
    ax.annotate(
        f"best {best['metrics/mAP50(B)']:.4f} @ {int(best['epoch'])}",
        (best["epoch"], best["metrics/mAP50(B)"]),
        textcoords="offset points", xytext=(-90, 12), fontsize=9, color="#c0392b",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_title("验证集精度曲线")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(df["epoch"], df["train/box_loss"], label="train box", color="#1f6f43", lw=1.6)
    ax.plot(df["epoch"], df["train/cls_loss"], label="train cls", color="#c8a24a", lw=1.6)
    ax.plot(df["epoch"], df["val/box_loss"], label="val box", color="#2c6ea8", lw=1.2, ls="--")
    ax.plot(df["epoch"], df["val/cls_loss"], label="val cls", color="#6d4c41", lw=1.2, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("训练 / 验证损失")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out)
    print("saved", args.out)
    print("final", df.iloc[-1][["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]].to_dict())
    print("best", best[["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]].to_dict())


if __name__ == "__main__":
    main()
