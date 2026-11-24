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
    """打开 feedback.db，并确保 events / runs 两张表存在。

    events 记用户行为（search/adopt/reject），runs 记每次匹配的完整结果快照。
    建表语句带 IF NOT EXISTS，所以每次调用都幂等。
    """
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(SCHEMA)
    conn.execute(RUN_SCHEMA)
    return conn


def _json_default(obj):
    """json.dumps(default=...) 用的兜底转换器。

    to_dict("records") 取出来的是 numpy 标量（int64/float64/bool_）而不是 Python
    原生类型，json 不认，统一用 .item() 降级成对应的基础类型。
    """
    # pandas/numpy 标量（如 int64）没有原生 JSON 类型，.item() 转成 Python 标量
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"无法序列化 {type(obj).__name__}")


def log_event(brief: str, action: str, daren_id: str | None = None, note: str | None = None):
    """往 events 表追加一条行为记录。

    Args:
        brief: 关联的投放需求原文，feedback_by_brief 按它回查。
        action: "search"（点了开始匹配）/ "adopt" / "reject"，stats 按此计数。
        daren_id: 采纳或不合适时传被操作的达人 ID，search 事件留 None。
        note: 预留的备注字段，目前未使用。
    """
    with _conn() as conn:
        conn.execute(
            "INSERT INTO events (ts, brief, daren_id, action, note) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), brief, daren_id, action, note),
        )


def save_run(brief: str, result: dict):
    """把一次 run_match 的结果整体序列化成 JSON 存进 runs 表。

    页面重新打开时靠 latest_run() 还原结果，所以这里存的必须能原样读回。
    """
    with _conn() as conn:
        conn.execute(
            "INSERT INTO runs (ts, brief, payload) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), brief,
             json.dumps(result, ensure_ascii=False, default=_json_default)),
        )


def latest_run() -> dict | None:
    """取最近一次匹配记录。

    Returns:
        {"brief": 输入原文, "result": run_match() 的返回值}；库里没有记录时返回 None。
    """
    with _conn() as conn:
        row = conn.execute("SELECT brief, payload FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return {"brief": row[0], "result": json.loads(row[1])} if row else None


def feedback_by_brief(brief: str) -> dict:
    """还原某条 brief 下已给过反馈的达人，key 统一用字符串 daren_id。

    Returns:
        {daren_id: "adopt" | "reject"}，界面据此把按钮换成已处理标记。
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT daren_id, action FROM events WHERE brief=? AND action IN ('adopt','reject')",
            (brief,),
        ).fetchall()
    return {daren_id: action for daren_id, action in rows}


def stats() -> dict:
    """全库累计计数，用于页面底部那行使用统计。

    Returns:
        {"searches": 匹配次数, "adoptions": 采纳次数, "rejections": 不合适次数}。
        注意云端磁盘是临时的，容器重启后归零。
    """
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events WHERE action='search'").fetchone()[0]
        adopted = conn.execute("SELECT COUNT(*) FROM events WHERE action='adopt'").fetchone()[0]
        rejected = conn.execute("SELECT COUNT(*) FROM events WHERE action='reject'").fetchone()[0]
    return {"searches": total, "adoptions": adopted, "rejections": rejected}
