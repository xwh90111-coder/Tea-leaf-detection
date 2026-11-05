"""Reset runtime data for a clean demo (tasks, detections, alerts, chats, files)."""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/tea_system.db"
BACKUP_DIR = ROOT / "data/backups"
BACKUP_DIR.mkdir(exist_ok=True)

if DB.exists():
    backup = BACKUP_DIR / f"tea_system_{datetime.now():%Y%m%d_%H%M%S}.db"
    shutil.copy2(DB, backup)
    print("backup:", backup)

con = sqlite3.connect(DB)
for table in ("chat_messages", "alerts", "detections", "tasks"):
    con.execute(f"DELETE FROM {table}")
con.commit()
con.execute("VACUUM")
con.close()
print("database reset")

for sub in ("uploads", "outputs", "reports"):
    d = ROOT / "data" / sub
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
        print("cleaned", d)
