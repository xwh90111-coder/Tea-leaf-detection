"""Final UI capture / demo seeding.

Runs the three detection modes through the browser, fills D1 with clean demo
records and writes the screenshots used in the report.

Usage:
    python scripts/final_capture.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
SHOTS = ROOT / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def static_sidebar(page):
    page.add_style_tag(content=".sidebar{position:static !important;height:auto !important;}")


def main():
    info = {}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel="msedge")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        pg.on("console", lambda m: print("console", m.type, m.text[:160]) if m.type == "error" else None)

        # ---------- single ----------
        pg.goto(f"{BASE}/detection", wait_until="networkidle")
        static_sidebar(pg)
        pg.set_input_files("#single-upload input[type=file]", str(ROOT / "data/samples/sample_severe.jpg"))
        pg.wait_for_selector("#single-preview img", timeout=30000)
        pg.click("#single-run")
        pg.wait_for_selector("#single-status:has-text('检测完成')", timeout=180000)
        time.sleep(5.5)
        pg.set_viewport_size({"width": 1440, "height": 1600})
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "01_detection_single.png"), full_page=True)
        status = pg.inner_text("#single-status")
        task_id = status.split("任务")[-1].strip().split()[0]
        info["single_task"] = task_id
        print("single:", status)

        # ---------- batch ----------
        pg.click("#tab-btn-batch")
        files = [
            str(ROOT / "data/samples/sample_batch_1.jpg"),
            str(ROOT / "data/samples/sample_batch_2.jpg"),
            str(ROOT / "data/samples/sample_mild.jpg"),
        ]
        pg.set_input_files("#batch-upload input[type=file]", files)
        pg.wait_for_selector("#batch-preview img", timeout=30000)
        pg.set_viewport_size({"width": 1440, "height": 1000})
        pg.click("#batch-run")
        pg.wait_for_selector("#batch-status:has-text('批量检测完成')", timeout=300000)
        time.sleep(5.5)
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "02_detection_batch.png"), full_page=True)
        print("batch:", pg.inner_text("#batch-status"))

        # ---------- video ----------
        pg.click("#tab-btn-video")
        pg.set_input_files("#video-upload input[type=file]", str(ROOT / "data/samples/tea_garden_demo.mp4"))
        pg.wait_for_selector("#video-preview video", timeout=30000)
        pg.click("#video-run")
        pg.wait_for_selector("#video-status:has-text('视频检测完成')", timeout=600000)
        time.sleep(5.5)
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "03_detection_video.png"), full_page=True)
        print("video:", pg.inner_text("#video-status"))

        # ---------- records ----------
        pg.set_viewport_size({"width": 1440, "height": 1100})
        pg.goto(f"{BASE}/records", wait_until="networkidle")
        pg.wait_for_selector("#query-table table", timeout=30000)
        time.sleep(2)
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "04_records.png"), full_page=True)
        print("records rows:", pg.locator("#query-table tbody tr").count())

        # ---------- assistant context ----------
        pg.goto(f"{BASE}/assistant?task={task_id}", wait_until="networkidle")
        pg.wait_for_timeout(2500)
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "05_assistant_context.png"), full_page=True)
        print("assistant sessions:", pg.locator(".session-btn").count())
        b.close()

    (ROOT / "reports" / "demo_tasks.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
