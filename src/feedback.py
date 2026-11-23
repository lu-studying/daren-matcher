import json
import sqlite3
from datetime import datetime

from src.config import FEEDBACK_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    brief TEXT NOT NULL,
    daren_id TEXT,
    action TEXT NOT NULL,
    note TEXT
);
"""

RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    brief TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _conn():
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(SCHEMA)
    conn.execute(RUN_SCHEMA)
    return conn


def _json_default(obj):
    # pandas/numpy 标量（如 int64）没有原生 JSON 类型，.item() 转成 Python 标量
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"无法序列化 {type(obj).__name__}")


def log_event(brief: str, action: str, daren_id: str | None = None, note: str | None = None):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO events (ts, brief, daren_id, action, note) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), brief, daren_id, action, note),
        )


def save_run(brief: str, result: dict):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO runs (ts, brief, payload) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), brief,
             json.dumps(result, ensure_ascii=False, default=_json_default)),
        )


def latest_run() -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT brief, payload FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return {"brief": row[0], "result": json.loads(row[1])} if row else None


def feedback_by_brief(brief: str) -> dict:
    """还原某条 brief 下已给过反馈的达人，key 统一用字符串 daren_id。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT daren_id, action FROM events WHERE brief=? AND action IN ('adopt','reject')",
            (brief,),
        ).fetchall()
    return {daren_id: action for daren_id, action in rows}


def stats() -> dict:
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events WHERE action='search'").fetchone()[0]
        adopted = conn.execute("SELECT COUNT(*) FROM events WHERE action='adopt'").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM events WHERE action='reject'").fetchone()[0]
    return {"searches": total, "adoptions": adopted, "rejections": rejected}
