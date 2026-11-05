"""Dataset audit: split ratio, class balance, label quality and leakage checks.

Usage:
    python src/dataset_report.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
CLASS_CN = {
    "algal-leaf-spot": "茶藻斑病",
    "brown-blight": "茶褐枯病",
    "grey-blight": "茶灰枯病",
}


def load_yaml() -> dict:
    with (ROOT / "data" / "tea.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    return {**cfg, "root": root}


def audit_split(root: Path, split: str, names: dict[int, str]) -> dict:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
    per_class = Counter()
    images_per_class = Counter()
    boxes_per_image = []
    sizes = Counter()
    w_norm, h_norm, area = [], [], []
    bad_labels = []
    missing = []
    for img in images:
        lbl = labels.get(img.stem)
        if lbl is None:
            missing.append(img.name)
            continue
        rows = [line.split() for line in lbl.read_text().splitlines() if line.strip()]
        boxes_per_image.append(len(rows))
        seen = set()
        for row in rows:
            if len(row) != 5:
                bad_labels.append(f"{lbl.name}: {row}")
                continue
            cid = int(float(row[0]))
            vals = [float(v) for v in row[1:]]
            if cid not in names:
                bad_labels.append(f"{lbl.name}: class {cid}")
                continue
            if not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_labels.append(f"{lbl.name}: invalid box {vals}")
                continue
            per_class[names[cid]] += 1
            seen.add(cid)
            w_norm.append(vals[2])
            h_norm.append(vals[3])
            area.append(vals[2] * vals[3])
        for cid in seen:
            images_per_class[names[cid]] += 1
        with Image.open(img) as im:
            sizes[im.size] += 1
    hashes = {}
    for img in images:
        hashes[img.name] = hashlib.md5(img.read_bytes()).hexdigest()
    return {
        "split": split,
        "images": len(images),
        "label_files": len(labels),
        "missing_labels": len(missing),
        "bad_labels": len(bad_labels),
        "bad_label_examples": bad_labels[:10],
        "instances": {k: int(v) for k, v in sorted(per_class.items())},
        "images_with_class": {k: int(v) for k, v in sorted(images_per_class.items())},
        "boxes_per_image_mean": round(float(np.mean(boxes_per_image)), 2) if boxes_per_image else 0,
        "boxes_per_image_max": int(np.max(boxes_per_image)) if boxes_per_image else 0,
        "empty_images": int(sum(1 for x in boxes_per_image if x == 0)),
        "image_sizes": {f"{k[0]}x{k[1]}": int(v) for k, v in sizes.most_common()},
        "box_width_mean": round(float(np.mean(w_norm)), 4) if w_norm else 0,
        "box_height_mean": round(float(np.mean(h_norm)), 4) if h_norm else 0,
        "box_area_mean": round(float(np.mean(area)), 5) if area else 0,
        "hashes": hashes,
    }


def main():
    cfg = load_yaml()
    names = {int(k): v for k, v in cfg["names"].items()}
    root = cfg["root"]
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "valid", "test"):
        splits[split] = audit_split(root, split, names)

    # duplicate images across splits (data leakage)
    seen = {}
    duplicates = []
    for split, info in splits.items():
        for name, h in info["hashes"].items():
            if h in seen:
                duplicates.append({"hash": h, "first": seen[h], "dup": f"{split}/{name}"})
            else:
                seen[h] = f"{split}/{name}"
        info.pop("hashes")

    total = sum(i["images"] for i in splits.values())
    for split, info in splits.items():
        info["ratio"] = round(info["images"] / total, 4)
    class_totals = Counter()
    for info in splits.values():
        class_totals.update(info["instances"])
    for split, info in splits.items():
        info["class_ratio"] = {
            k: round(info["instances"].get(k, 0) / class_totals[k], 4) if class_totals[k] else 0
            for k in class_totals
        }

    payload = {
        "dataset_root": str(root),
        "classes": [names[k] for k in sorted(names)],
        "total_images": total,
        "split_totals": {k: v["images"] for k, v in splits.items()},
        "class_totals": dict(class_totals),
        "duplicates_across_splits": duplicates,
        "splits": splits,
    }
    (REPORTS / "dataset_stats.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    labels = [CLASS_CN.get(names[k], names[k]) for k in sorted(names)]
    width = 0.26
    for i, split in enumerate(("train", "valid", "test")):
        vals = [splits[split]["instances"].get(names[k], 0) for k in sorted(names)]
        axes[0].bar(np.arange(len(labels)) + (i - 1) * width, vals, width, label=split)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("实例数")
    axes[0].set_title("各数据划分的类别实例数")
    axes[0].legend()
    axes[1].bar(
        [CLASS_CN.get(names[k], names[k]) for k in sorted(names)],
        [class_totals[names[k]] for k in sorted(names)],
        color=["#2e7d32", "#c62828", "#6d4c41"],
    )
    axes[1].set_title("数据集类别总量")
    axes[1].set_ylabel("实例数")
    fig.tight_layout()
    fig.savefig(FIGS / "dataset_distribution.png")
    plt.close(fig)

    print(json.dumps({k: v for k, v in payload.items() if k != "splits"}, ensure_ascii=False, indent=2))
    for split in ("train", "valid", "test"):
        s = splits[split]
        print(split, s["images"], s["ratio"], s["instances"], "missing", s["missing_labels"], "bad", s["bad_labels"])


if __name__ == "__main__":
    main()
