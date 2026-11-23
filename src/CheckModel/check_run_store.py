import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import FEEDBACK_DB
from src.feedback import latest_run, save_run

FAKE_BRIEF = "自检用假数据（跑完会自动删除）"

fake = {
    "brief": FAKE_BRIEF,
    "brief_dict": {"count": np.int64(5)},
    "fallback": np.bool_(True),
    "recall_count": np.int64(12),
    "results": [
        {
            "daren_id": np.int64(7),
            "name": "测试达人",
            "profile_text": "x",
            "score": np.int64(88),
            "reason": "r",
            "risk": None,
        }
    ],
}

save_run(FAKE_BRIEF, fake)
back = latest_run()
print("latest_run 取回的 brief =", back["brief"])
row = back["result"]["results"][0]
print("daren_id  序列化后类型 =", type(row["daren_id"]).__name__)
print("score     序列化后类型 =", type(row["score"]).__name__)
print("fallback  序列化后类型 =", type(back["result"]["fallback"]).__name__)
print("risk      序列化后值   =", row["risk"])

with sqlite3.connect(FEEDBACK_DB) as conn:
    conn.execute("DELETE FROM runs WHERE brief=?", (FAKE_BRIEF,))
print("自检数据已清理")
