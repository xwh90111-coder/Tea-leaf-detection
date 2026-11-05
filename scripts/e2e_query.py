"""E2E test for the data-query page (filters, charts, pagination, Excel export)."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
SHOTS = ROOT / "docs" / "screenshots"
DOWNLOADS = ROOT / "data" / "reports"


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel="msedge")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        pg.on("console", lambda m: print("console", m.type, m.text[:180]) if m.type == "error" else None)
        pg.goto(f"{BASE}/records", wait_until="networkidle")
        pg.wait_for_selector("#query-table table", timeout=30000)
        time.sleep(1.5)
        pg.screenshot(path=str(SHOTS / "records.png"), full_page=True)
        print("rows:", pg.locator("#query-table tbody tr").count())
        print("pageinfo:", pg.inner_text("#query-pageinfo"))

        # filter to one class and apply
        pg.click("#query-classes")
        time.sleep(0.4)
        pg.keyboard.press("ArrowDown")
        pg.keyboard.press("Enter")
        pg.keyboard.press("Escape")
        pg.click("#query-apply")
        time.sleep(2)
        print("after filter:", pg.inner_text("#query-pageinfo"))
        pg.screenshot(path=str(SHOTS / "records_filtered.png"), full_page=True)

        # export
        with pg.expect_download(timeout=60000) as dl:
            pg.click("#query-export")
        download = dl.value
        out = DOWNLOADS / (download.suggested_filename or "export.xlsx")
        download.save_as(str(out))
        print("downloaded:", out, out.stat().st_size)
        b.close()


if __name__ == "__main__":
    main()
