"""Capture assistant Q&A and report screenshots from seeded sessions."""
from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:8050"


def static_sidebar(page):
    page.add_style_tag(content=".sidebar{position:static !important;height:auto !important;}")
    page.add_style_tag(content=".chat-window{max-height:none !important;}")


def main():
    task_id = json.loads((ROOT / "reports/demo_tasks.json").read_text(encoding="utf-8"))["single_task"]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel="msedge")
        pg = b.new_page(viewport={"width": 1440, "height": 1100})
        pg.on("console", lambda m: print("console", m.type, m.text[:160]) if m.type == "error" else None)
        pg.goto(f"{BASE}/assistant?task={task_id}", wait_until="networkidle")
        pg.wait_for_timeout(2500)
        n = pg.locator(".session-btn").count()
        print("sessions", n)
        static_sidebar(pg)
        pg.screenshot(path=str(SHOTS / "05_assistant_context.png"), full_page=True)

        # newest session = report, second = Q&A
        if n >= 1:
            pg.locator(".session-btn").nth(0).click()
            pg.wait_for_timeout(2500)
            pg.set_viewport_size({"width": 1440, "height": 1400})
            static_sidebar(pg)
            pg.screenshot(path=str(SHOTS / "07_assistant_report.png"), full_page=True)
            print("report bubbles", pg.locator("#assistant-chat .bubble").count())

        if n >= 2:
            pg.locator(".session-btn").nth(1).click()
            pg.wait_for_timeout(2500)
            pg.set_viewport_size({"width": 1440, "height": 1400})
            static_sidebar(pg)
            pg.screenshot(path=str(SHOTS / "06_assistant_answer.png"), full_page=True)
            print("answer bubbles", pg.locator("#assistant-chat .bubble").count())
            print("sources", pg.locator("#assistant-sources .source-item").count())
        b.close()


if __name__ == "__main__":
    main()
