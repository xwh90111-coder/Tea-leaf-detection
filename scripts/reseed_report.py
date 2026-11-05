from __future__ import annotations
import json, time, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8050"
task_id = json.loads((ROOT / "reports/demo_tasks.json").read_text(encoding="utf-8"))["single_task"]
deps = requests.get(f"{BASE}/_dash-dependencies", timeout=10).json()
key = next(d["output"] for d in deps if isinstance(d.get("output"), str) and "assistant-report-store" in d["output"])
payload = {
    "output": key,
    "outputs": [
        {"id": "assistant-chat", "property": "children"},
        {"id": "assistant-history", "property": "data"},
        {"id": "assistant-sources", "property": "children"},
        {"id": "assistant-report-store", "property": "data"},
        {"id": "assistant-download", "property": "data"},
    ],
    "inputs": [{"id": "assistant-report", "property": "n_clicks", "value": 1}],
    "state": [
        {"id": "assistant-task", "property": "value", "value": None},
        {"id": "assistant-active-task", "property": "data", "value": task_id},
        {"id": "assistant-history", "property": "data", "value": []},
        {"id": "assistant-session", "property": "data", "value": None},
    ],
    "changedPropIds": ["assistant-report.n_clicks"],
}
t0 = time.time()
r = requests.post(f"{BASE}/_dash-update-component", json=payload, timeout=900)
print("status", r.status_code, "elapsed", round(time.time() - t0, 1), flush=True)
