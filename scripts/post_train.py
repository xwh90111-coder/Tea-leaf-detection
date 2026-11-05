"""Wait for training to finish, then run the final evaluation pipeline.

Steps: strip optimizer -> evaluate test -> benchmark -> training curves ->
reproduction report.  Safe to start while training is still running.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs/detect/yolo12n_tea_400/results.csv"
BEST = ROOT / "runs/detect/yolo12n_tea_400/weights/best.pt"
DEPLOY = ROOT / "models/best.pt"
LOG = ROOT / "reports/post_train.log"
LOG.parent.mkdir(exist_ok=True)


def log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def last_epoch() -> int:
    try:
        with CSV.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return int(float(rows[-1][0]))
    except Exception:
        return 0


def run(cmd: list[str], timeout: int | None = None) -> int:
    log("RUN " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-800:]
    log(f"exit={p.returncode} elapsed={elapsed:.1f}s\n{tail}")
    return p.returncode


def main():
    log("post-train watcher started")
    while True:
        epoch = last_epoch()
        age = time.time() - CSV.stat().st_mtime if CSV.exists() else 1e9
        if epoch >= 400:
            log(f"epoch {epoch} reached")
            break
        if epoch > 50 and age > 300:
            log(f"training stopped early at epoch {epoch} (csv idle {age:.0f}s)")
            break
        log(f"waiting... epoch={epoch} idle={age:.0f}s")
        time.sleep(30)

    time.sleep(20)
    log("finalizing")

    shutil.copy2(BEST, DEPLOY)
    from ultralytics.utils.torch_utils import strip_optimizer

    strip_optimizer(str(DEPLOY))
    log(f"deployed {DEPLOY} ({DEPLOY.stat().st_size / 1024 / 1024:.2f} MB)")

    run([sys.executable, "src/evaluate.py", "--weights", str(DEPLOY), "--split", "test", "--name", "test_final"])
    run([sys.executable, "src/benchmark.py", "--weights", str(DEPLOY), "--runs", "200", "--runs-wall", "100"])
    run([sys.executable, "src/plot_training.py"])
    run([sys.executable, "scripts/build_report.py"])

    (ROOT / "reports/POST_TRAIN_DONE").write_text(datetime.now().isoformat(), encoding="utf-8")
    log("ALL DONE")


if __name__ == "__main__":
    main()
