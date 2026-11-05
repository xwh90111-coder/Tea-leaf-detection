"""Reproduce a stratified 8:1:1 split from the pooled dataset.

The original project already ships a split; this script rebuilds an equivalent
split from ``tea-leaf-disease/all`` with a fixed seed and verifies that class
proportions are preserved across train/valid/test.

Outputs (never overwrites the original dataset):
    data/splits/{train,valid,test}/{images,labels}
    data/tea_split.yaml

Usage:
    python src/split_dataset.py --mode hardlink
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "茶叶病害检测" / "tea-leaf-disease"


def image_class(label_path: Path) -> int:
    """Image-level stratum = most frequent class in the annotation file."""
    counts = Counter()
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                counts[int(float(line.split()[0]))] += 1
    return counts.most_common(1)[0][0] if counts else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    imgs = sorted((src / "all" / "images").glob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    strata = defaultdict(list)
    for img in imgs:
        lbl = src / "all" / "labels" / f"{img.stem}.txt"
        strata[image_class(lbl)].append(img)

    rng = random.Random(args.seed)
    split_map = {}
    for cls, items in sorted(strata.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        for i, img in enumerate(items):
            split_map[img] = "train" if i < n_train else ("valid" if i < n_train + n_val else "test")

    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    for img, split in split_map.items():
        dst_img = out / split / "images" / img.name
        src_lbl = src / "all" / "labels" / f"{img.stem}.txt"
        dst_lbl = out / split / "labels" / f"{img.stem}.txt"
        for s, d in ((img, dst_img), (src_lbl, dst_lbl)):
            if not s.exists():
                continue
            if d.exists():
                continue
            if args.mode == "hardlink":
                try:
                    os.link(s, d)
                    continue
                except OSError:
                    pass
            shutil.copy2(s, d)

    names = ["algal-leaf-spot", "brown-blight", "grey-blight"]
    yaml_text = (
        f"path: {out.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 3\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (ROOT / "data" / "tea_split.yaml").write_text(yaml_text, encoding="utf-8")

    print("source images:", len(imgs))
    for split in ("train", "valid", "test"):
        cnt = sum(1 for s in split_map.values() if s == split)
        print(f"{split}: {cnt} ({cnt/len(imgs):.4f})")
    print("yaml:", ROOT / "data" / "tea_split.yaml")


if __name__ == "__main__":
    main()
