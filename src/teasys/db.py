"""D1 - structured disease database (SQLite).

Every detection request creates one ``task`` row (one image / one video / one
file inside a batch) and one ``detection`` row per predicted lesion.  Alert
records and assistant chat history are stored in the same relational database,
which maps directly onto data stores D1 in the paper's DFD.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    batch_id        TEXT,
    task_type       TEXT NOT NULL,
    source_name     TEXT,
    terminal_id     TEXT,
    total_objects   INTEGER DEFAULT 0,
    class_counts    TEXT DEFAULT '{}',
    max_confidence  REAL DEFAULT 0,
    avg_confidence  REAL DEFAULT 0,
    severity        TEXT DEFAULT '无',
    alert_level     TEXT DEFAULT 'none',
    alert_message   TEXT,
    elapsed_ms      REAL DEFAULT 0,
    annotated_path  TEXT,
    media_path      TEXT,
    status          TEXT DEFAULT 'done',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_batch ON tasks(batch_id);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    class_id    INTEGER,
    class_name  TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    area_ratio  REAL DEFAULT 0,
    created_at  TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE INDEX IF NOT EXISTS idx_det_task ON detections(task_id);
CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_det_created ON detections(created_at);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    level       TEXT,
    disease     TEXT,
    message     TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    role        TEXT,
    content     TEXT,
    sources     TEXT,
    task_id     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with _LOCK, _connect() as con:
        con.executescript(_SCHEMA)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tasks / detections
# ---------------------------------------------------------------------------
def create_task(
    task_type: str,
    source_name: str,
    terminal_id: str | None = None,
    batch_id: str | None = None,
    media_path: str | None = None,
) -> str:
    task_id = uuid.uuid4().hex[:12]
    with _LOCK, _connect() as con:
        con.execute(
            "INSERT INTO tasks (id, batch_id, task_type, source_name, terminal_id,"
            " media_path, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                task_id,
                batch_id,
                task_type,
                source_name,
                terminal_id or config.TERMINAL_ID,
                media_path,
                "running",
                now(),
            ),
        )
    return task_id


def add_detections(task_id: str, detections: Sequence[dict[str, Any]]) -> None:
    if not detections:
        return
    rows = [
        (
            task_id,
            int(d.get("class_id", -1)),
            d.get("class_name", ""),
            float(d.get("confidence", 0)),
            float(d["bbox"][0]),
            float(d["bbox"][1]),
            float(d["bbox"][2]),
            float(d["bbox"][3]),
            float(d.get("area_ratio", 0)),
            now(),
        )
        for d in detections
    ]
    with _LOCK, _connect() as con:
        con.executemany(
            "INSERT INTO detections (task_id, class_id, class_name, confidence,"
            " x1, y1, x2, y2, area_ratio, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


def finalize_task(
    task_id: str,
    *,
    annotated_path: str | None = None,
    elapsed_ms: float = 0,
    severity: str = "无",
    alert_level: str = "none",
    alert_message: str | None = None,
) -> dict[str, Any]:
    with _LOCK, _connect() as con:
        dets = con.execute(
            "SELECT class_name, confidence, area_ratio FROM detections WHERE task_id=?",
            (task_id,),
        ).fetchall()
        counts: dict[str, int] = {}
        confs: list[float] = []
        for r in dets:
            counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
            confs.append(float(r["confidence"] or 0))
        con.execute(
            "UPDATE tasks SET total_objects=?, class_counts=?, max_confidence=?,"
            " avg_confidence=?, severity=?, alert_level=?, alert_message=?,"
            " elapsed_ms=?, annotated_path=?, status='done' WHERE id=?",
            (
                len(dets),
                json.dumps(counts, ensure_ascii=False),
                max(confs) if confs else 0,
                sum(confs) / len(confs) if confs else 0,
                severity,
                alert_level,
                alert_message,
                elapsed_ms,
                annotated_path,
                task_id,
            ),
        )
        if alert_level in ("info", "warning", "danger") and alert_message:
            con.execute(
                "INSERT INTO alerts (task_id, level, disease, message, created_at)"
                " VALUES (?,?,?,?,?)",
                (
                    task_id,
                    alert_level,
                    ",".join(counts.keys()),
                    alert_message,
                    now(),
                ),
            )
    return get_task(task_id) or {}


def get_task(task_id: str) -> dict[str, Any] | None:
    with _LOCK, _connect() as con:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return None
        task = dict(row)
        task["class_counts"] = json.loads(task["class_counts"] or "{}")
        task["detections"] = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM detections WHERE task_id=? ORDER BY confidence DESC",
                (task_id,),
            ).fetchall()
        ]
    return task


def recent_tasks(limit: int = 8) -> list[dict[str, Any]]:
    with _LOCK, _connect() as con:
        rows = con.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        t = dict(r)
        t["class_counts"] = json.loads(t["class_counts"] or "{}")
        out.append(t)
    return out


def tasks_by_batch(batch_id: str) -> list[dict[str, Any]]:
    with _LOCK, _connect() as con:
        rows = con.execute(
            "SELECT * FROM tasks WHERE batch_id=? ORDER BY created_at, rowid",
            (batch_id,),
        ).fetchall()
    out = []
    for r in rows:
        t = dict(r)
        t["class_counts"] = json.loads(t["class_counts"] or "{}")
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Queries / statistics
# ---------------------------------------------------------------------------
def _where(
    date_from: str | None,
    date_to: str | None,
    classes: Sequence[str] | None,
    terminals: Sequence[str] | None,
    task_types: Sequence[str] | None,
    min_conf: float | None,
    keyword: str | None,
) -> tuple[str, list[Any]]:
    clauses, params = [], []
    if date_from:
        clauses.append("t.created_at >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        clauses.append("t.created_at <= ?")
        params.append(f"{date_to} 23:59:59")
    if terminals:
        clauses.append("t.terminal_id IN (%s)" % ",".join("?" * len(terminals)))
        params.extend(terminals)
    if task_types:
        clauses.append("t.task_type IN (%s)" % ",".join("?" * len(task_types)))
        params.extend(task_types)
    if min_conf is not None:
        clauses.append("t.max_confidence >= ?")
        params.append(min_conf)
    if keyword:
        clauses.append("(t.source_name LIKE ? OR t.id LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if classes:
        placeholders = ",".join("?" * len(classes))
        clauses.append(
            f"EXISTS (SELECT 1 FROM detections d WHERE d.task_id=t.id"
            f" AND d.class_name IN ({placeholders}))"
        )
        params.extend(classes)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def query_tasks(
    date_from: str | None = None,
    date_to: str | None = None,
    classes: Sequence[str] | None = None,
    terminals: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    min_conf: float | None = None,
    keyword: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    where, params = _where(
        date_from, date_to, classes, terminals, task_types, min_conf, keyword
    )
    with _LOCK, _connect() as con:
        total = con.execute(f"SELECT COUNT(*) c FROM tasks t{where}", params).fetchone()["c"]
        rows = con.execute(
            f"SELECT t.* FROM tasks t{where} ORDER BY t.created_at DESC, t.rowid DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    out = []
    for r in rows:
        t = dict(r)
        t["class_counts"] = json.loads(t["class_counts"] or "{}")
        out.append(t)
    return out, total


def summary(
    date_from: str | None = None,
    date_to: str | None = None,
    classes: Sequence[str] | None = None,
    terminals: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    min_conf: float | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    where, params = _where(
        date_from, date_to, classes, terminals, task_types, min_conf, keyword
    )
    with _LOCK, _connect() as con:
        row = con.execute(
            f"SELECT COUNT(*) tasks, COALESCE(SUM(t.total_objects),0) objects,"
            f" COALESCE(AVG(t.avg_confidence),0) avg_conf,"
            f" COALESCE(SUM(CASE WHEN t.alert_level='danger' THEN 1 ELSE 0 END),0) danger"
            f" FROM tasks t{where}",
            params,
        ).fetchone()
        today = datetime.now().strftime("%Y-%m-%d")
        today_tasks = con.execute(
            "SELECT COUNT(*) c FROM tasks WHERE created_at LIKE ?", (f"{today}%",)
        ).fetchone()["c"]
        unacked = con.execute(
            "SELECT COUNT(*) c FROM alerts WHERE acknowledged=0"
        ).fetchone()["c"]
    return {
        "tasks": row["tasks"],
        "objects": row["objects"],
        "avg_confidence": round(float(row["avg_conf"] or 0), 4),
        "danger": row["danger"],
        "today_tasks": today_tasks,
        "unacked_alerts": unacked,
    }


def class_distribution(
    date_from: str | None = None,
    date_to: str | None = None,
    classes: Sequence[str] | None = None,
    terminals: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    min_conf: float | None = None,
    keyword: str | None = None,
) -> list[dict[str, Any]]:
    where, params = _where(
        date_from, date_to, classes, terminals, task_types, min_conf, keyword
    )
    sql = (
        "SELECT d.class_name, COUNT(*) c, AVG(d.confidence) conf FROM detections d"
        " JOIN tasks t ON t.id=d.task_id"
        f"{where} GROUP BY d.class_name ORDER BY c DESC"
    )
    with _LOCK, _connect() as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def daily_trend(
    days: int = 14,
    classes: Sequence[str] | None = None,
    terminals: Sequence[str] | None = None,
    min_conf: float | None = None,
) -> list[dict[str, Any]]:
    start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    where, params = _where(start, None, classes, terminals, None, min_conf, None)
    with _LOCK, _connect() as con:
        rows = con.execute(
            f"SELECT substr(t.created_at,1,10) d, COUNT(*) tasks,"
            f" COALESCE(SUM(t.total_objects),0) objects FROM tasks t{where}"
            f" GROUP BY substr(t.created_at,1,10) ORDER BY d",
            params,
        ).fetchall()
    data = {r["d"]: {"date": r["d"], "tasks": r["tasks"], "objects": r["objects"]} for r in rows}
    out = []
    for i in range(days):
        d = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        out.append(data.get(d, {"date": d, "tasks": 0, "objects": 0}))
    return out


def terminal_distribution(
    date_from: str | None = None,
    date_to: str | None = None,
    classes: Sequence[str] | None = None,
    min_conf: float | None = None,
) -> list[dict[str, Any]]:
    where, params = _where(date_from, date_to, classes, None, None, min_conf, None)
    with _LOCK, _connect() as con:
        rows = con.execute(
            f"SELECT t.terminal_id, COUNT(*) c FROM tasks t{where}"
            " GROUP BY t.terminal_id ORDER BY c DESC",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def confidence_histogram(
    bins: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
    classes: Sequence[str] | None = None,
    terminals: Sequence[str] | None = None,
    min_conf: float | None = None,
) -> list[dict[str, Any]]:
    where, params = _where(date_from, date_to, classes, terminals, None, min_conf, None)
    with _LOCK, _connect() as con:
        rows = con.execute(
            f"SELECT d.confidence FROM detections d JOIN tasks t ON t.id=d.task_id{where}",
            params,
        ).fetchall()
    import numpy as np

    vals = np.array([r["confidence"] for r in rows], dtype=float)
    if vals.size == 0:
        return []
    hist, edges = np.histogram(vals, bins=bins, range=(0, 1))
    return [
        {"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(hist[i])}
        for i in range(len(hist))
    ]


def list_alerts(limit: int = 20, unacked_only: bool = False) -> list[dict[str, Any]]:
    sql = "SELECT a.*, t.source_name FROM alerts a LEFT JOIN tasks t ON t.id=a.task_id"
    if unacked_only:
        sql += " WHERE a.acknowledged=0"
    sql += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
    with _LOCK, _connect() as con:
        return [dict(r) for r in con.execute(sql, (limit,)).fetchall()]


def acknowledge_alert(alert_id: int) -> None:
    with _LOCK, _connect() as con:
        con.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))


# ---------------------------------------------------------------------------
# Assistant history
# ---------------------------------------------------------------------------
def save_message(
    session_id: str, role: str, content: str, sources: str = "", task_id: str = ""
) -> None:
    with _LOCK, _connect() as con:
        con.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, task_id, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (session_id, role, content, sources, task_id, now()),
        )


def chat_history(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK, _connect() as con:
        rows = con.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def all_chat_sessions(limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK, _connect() as con:
        rows = con.execute(
            "SELECT session_id, MIN(created_at) started, COUNT(*) msgs,"
            " (SELECT content FROM chat_messages c2 WHERE c2.session_id=c1.session_id"
            "  AND role='user' ORDER BY id LIMIT 1) first_q"
            " FROM chat_messages c1 GROUP BY session_id ORDER BY started DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def export_excel(path: Path, **filters: Any) -> Path:
    import pandas as pd

    tasks, _ = query_tasks(**{**filters, "limit": 100000, "offset": 0})
    task_rows = []
    for t in tasks:
        task_rows.append(
            {
                "任务ID": t["id"],
                "时间": t["created_at"],
                "类型": {"single": "单张图像", "batch": "批量图像", "video": "视频流"}.get(
                    t["task_type"], t["task_type"]
                ),
                "文件": t["source_name"],
                "终端": t["terminal_id"],
                "病灶数": t["total_objects"],
                "类别统计": json.dumps(t["class_counts"], ensure_ascii=False),
                "最高置信度": round(t["max_confidence"], 4),
                "严重程度": t["severity"],
                "预警等级": t["alert_level"],
                "耗时(ms)": round(t["elapsed_ms"], 1),
            }
        )
    detail_rows = []
    with _LOCK, _connect() as con:
        for t in tasks:
            for d in con.execute(
                "SELECT * FROM detections WHERE task_id=? ORDER BY confidence DESC",
                (t["id"],),
            ).fetchall():
                detail_rows.append(
                    {
                        "任务ID": t["id"],
                        "文件": t["source_name"],
                        "时间": t["created_at"],
                        "类别": d["class_name"],
                        "置信度": round(d["confidence"], 4),
                        "x1": round(d["x1"], 1),
                        "y1": round(d["y1"], 1),
                        "x2": round(d["x2"], 1),
                        "y2": round(d["y2"], 1),
                        "面积占比": round(d["area_ratio"], 5),
                    }
                )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(task_rows).to_excel(writer, sheet_name="任务汇总", index=False)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="检测明细", index=False)
    return path
