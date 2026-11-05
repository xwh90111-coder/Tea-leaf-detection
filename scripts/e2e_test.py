"""End-to-end UI test with Playwright (uses the installed Edge browser).

Usage:
    python scripts/e2e_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
SHOTS = ROOT / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def shot(page, name: str, full: bool = True):
    page.screenshot(path=str(SHOTS / name), full_page=full)
    print("screenshot:", name)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda m: print("console:", m.type, m.text[:200]) if m.type == "error" else None)

        # ---- single detection -----------------------------------------
        page.goto(f"{BASE}/detection", wait_until="networkidle")
        page.set_input_files("#single-upload input[type=file]", str(ROOT / "data/samples/sample_severe.jpg"))
        page.wait_for_selector("#single-preview img", timeout=30000)
        page.click("#single-run")
        page.wait_for_selector("#single-status:has-text('检测完成')", timeout=120000)
        time.sleep(1.5)
        shot(page, "detection_result.png")
        status = page.inner_text("#single-status")
        print("STATUS:", status)
        task_id = status.split("任务")[-1].strip().split()[0]
        print("TASK:", task_id)

        # ---- assistant with the task ----------------------------------
        page.goto(f"{BASE}/assistant?task={task_id}", wait_until="networkidle")
        page.wait_for_selector("#assistant-task-summary .status-line", timeout=30000)
        time.sleep(1)
        shot(page, "assistant_task.png")
        page.fill("#assistant-input", "这株茶树的病害严重吗？应该采取什么防治措施？")
        page.click("#assistant-send")
        page.wait_for_function(
            "() => document.querySelectorAll('#assistant-chat .bubble').length >= 2",
            timeout=300000,
        )
        time.sleep(1.5)
        shot(page, "assistant_answer.png")

        # ---- records ---------------------------------------------------
        page.goto(f"{BASE}/records", wait_until="networkidle")
        page.wait_for_selector("#query-table table", timeout=30000)
        time.sleep(1.2)
        shot(page, "records.png")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
