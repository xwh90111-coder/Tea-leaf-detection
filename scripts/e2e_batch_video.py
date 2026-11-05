"""E2E tests for batch and video detection pages."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
SHOTS = ROOT / "docs" / "screenshots"


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel="msedge")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        pg.on("console", lambda m: print("console", m.type, m.text[:180]) if m.type == "error" else None)
        pg.goto(f"{BASE}/detection", wait_until="networkidle")

        # batch
        pg.click("#tab-btn-batch")
        files = [str(ROOT / f"data/samples/sample_batch_{i}.jpg") for i in (1, 2)]
        files.append(str(ROOT / "data/samples/sample_mild.jpg"))
        pg.set_input_files("#batch-upload input[type=file]", files)
        pg.wait_for_selector("#batch-preview img", timeout=30000)
        pg.click("#batch-run")
        pg.wait_for_selector("#batch-status:has-text('批量检测完成')", timeout=300000)
        time.sleep(2)
        pg.screenshot(path=str(SHOTS / "batch_result.png"), full_page=True)
        print("BATCH:", pg.inner_text("#batch-status"))

        # video
        pg.click("#tab-btn-video")
        pg.set_input_files("#video-upload input[type=file]", str(ROOT / "data/samples/tea_garden_demo.mp4"))
        pg.wait_for_selector("#video-preview video", timeout=30000)
        pg.click("#video-run")
        pg.wait_for_selector("#video-status:has-text('视频检测完成')", timeout=600000)
        time.sleep(3)
        pg.screenshot(path=str(SHOTS / "video_result.png"), full_page=True)
        print("VIDEO:", pg.inner_text("#video-status"))
        b.close()


if __name__ == "__main__":
    main()
