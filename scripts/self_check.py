"""Self-check for the reproduced system (no GPU inference required).

Usage:
    python scripts/self_check.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teasys import config, db
from teasys.knowledge import kb_meta
from teasys.llm import llm_status


def ok(label: str, cond: bool, detail: str = ""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    checks = []
    checks.append(ok("数据集配置", (ROOT / "data/tea.yaml").exists()))
    checks.append(ok("训练结果", (ROOT / "runs/detect/yolo12n_tea_400/results.csv").exists()))
    weights = ROOT / "runs/detect/yolo12n_tea_400/weights/best.pt"
    checks.append(ok("复现权重", weights.exists(), str(weights)))
    checks.append(ok("部署权重 models/best.pt", (ROOT / "models/best.pt").exists()))

    db.init_db()
    con = sqlite3.connect(config.DB_PATH)
    n_tasks = con.execute("select count(*) from tasks").fetchone()[0]
    n_dets = con.execute("select count(*) from detections").fetchone()[0]
    checks.append(ok("D1 数据库（任务/病灶）", n_tasks >= 1 and n_dets >= 1, f"{n_tasks} / {n_dets}"))

    kb = kb_meta()
    checks.append(ok("D2 知识库索引", kb.get("chunks", 0) >= 30, f"{kb.get('chunks')} 片段 / {kb.get('docs')} 篇"))

    llm = llm_status()
    checks.append(ok("本地大模型文件", llm.get("available", False), llm.get("detail", "")))
    checks.append(ok("向量模型 BGE", (ROOT / "models/bge-small-zh-v1.5/config.json").exists()))
    checks.append(ok("DeepSeek GGUF", (ROOT / "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf").exists()))

    metrics = ROOT / "reports/metrics_test.json"
    bench = ROOT / "reports/benchmark.json"
    checks.append(ok("测试集指标", metrics.exists()))
    checks.append(ok("速度基准", bench.exists()))
    if metrics.exists():
        m = json.loads(metrics.read_text(encoding="utf-8"))
        print("   mAP50 =", round(m["map_mode"]["mAP50"], 4),
              " mAP50-95 =", round(m["map_mode"]["mAP50_95"], 4))
    if bench.exists():
        b = json.loads(bench.read_text(encoding="utf-8"))
        print("   params =", b.get("parameters"), " GFLOPs =", b.get("gflops"),
              " FPS(e2e) =", b.get("fps_end2end"))

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
