"""Test the RAG assistant end-to-end (outside Dash) and save the answer."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teasys import db
from teasys.assistant import answer_question

con = sqlite3.connect(ROOT / "data/tea_system.db")
con.row_factory = sqlite3.Row
task_id = con.execute(
    "select id from tasks where task_type='single' order by created_at desc limit 1"
).fetchone()["id"]
task = db.get_task(task_id)
print("task", task_id, task["class_counts"], flush=True)
t0 = time.time()
res = answer_question(
    "这株茶树病害严重吗？请给出诊断、严重程度判断和可执行的防治建议。",
    task=task,
    history=[],
)
elapsed = time.time() - t0
out = ROOT / "reports" / "llm_test_answer.md"
out.write_text(
    f"# RAG assistant test\n\n- task: {task_id}\n- elapsed: {elapsed:.1f}s\n"
    f"- model: {res['model']}\n- fallback: {res['fallback']}\n"
    f"- sources: {json.dumps([s['source'] for s in res['sources']], ensure_ascii=False)}\n\n"
    f"{res['answer']}\n",
    encoding="utf-8",
)
print("elapsed", round(elapsed, 1), "chars", len(res["answer"]), "fallback", res["fallback"], flush=True)
print(res["answer"][:300], flush=True)
