"""Seed a genuine DeepSeek-R1-14B Q&A session and a full report for the demo DB.

Run after scripts/final_capture.py (uses reports/demo_tasks.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
task_id = json.loads((ROOT / "reports/demo_tasks.json").read_text(encoding="utf-8"))["single_task"]
print("task", task_id, flush=True)

deps = requests.get(f"{BASE}/_dash-dependencies", timeout=10).json()


def find_output(fragment: str) -> str:
    for dep in deps:
        out = dep.get("output")
        if isinstance(out, str) and fragment in out:
            return out
    raise RuntimeError(f"callback not found: {fragment}")


def post(output: str, outputs: list[dict], inputs: list[dict], state: list[dict], timeout=900):
    payload = {
        "output": output,
        "outputs": outputs,
        "inputs": inputs,
        "state": state,
        "changedPropIds": [f"{inputs[0]['id']}.{inputs[0]['property']}"],
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/_dash-update-component", json=payload, timeout=timeout)
    print("status", r.status_code, "elapsed", round(time.time() - t0, 1), "bytes", len(r.text), flush=True)
    return r


# ---- Q&A ---------------------------------------------------------------
send_key = find_output("assistant-session")
post(
    send_key,
    [
        {"id": "assistant-chat", "property": "children"},
        {"id": "assistant-history", "property": "data"},
        {"id": "assistant-sources", "property": "children"},
        {"id": "assistant-input", "property": "value"},
        {"id": "assistant-session", "property": "data"},
    ],
    [{"id": "assistant-send", "property": "n_clicks", "value": 1}],
    [
        {"id": "assistant-input", "property": "value",
         "value": "这株茶树病害严重吗？请给出诊断、严重程度判断和可执行的防治建议。"},
        {"id": "assistant-task", "property": "value", "value": None},
        {"id": "assistant-active-task", "property": "data", "value": task_id},
        {"id": "assistant-history", "property": "data", "value": []},
        {"id": "assistant-session", "property": "data", "value": None},
    ],
)

# ---- report ------------------------------------------------------------
report_key = find_output("assistant-report-store")
post(
    report_key,
    [
        {"id": "assistant-chat", "property": "children"},
        {"id": "assistant-history", "property": "data"},
        {"id": "assistant-sources", "property": "children"},
        {"id": "assistant-report-store", "property": "data"},
        {"id": "assistant-download", "property": "data"},
    ],
    [{"id": "assistant-report", "property": "n_clicks", "value": 1}],
    [
        {"id": "assistant-task", "property": "value", "value": None},
        {"id": "assistant-active-task", "property": "data", "value": task_id},
        {"id": "assistant-history", "property": "data", "value": []},
        {"id": "assistant-session", "property": "data", "value": None},
    ],
)
print("done", flush=True)
