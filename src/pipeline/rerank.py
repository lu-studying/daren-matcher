from src.config import RERANK_TOP_N
from src.llm import chat_json

SYSTEM = """你是资深达人投放顾问。根据投放需求评估一个达人候选，输出JSON：
{"score": 0-100整数, "reason": "为什么匹配，一句话，引用具体数据", "risk": "最主要的一个风险点，没有则填 null"}
评分依据：品类契合 > 人群与风格契合 > 量级与预算合理性。宁低勿高，80分以上必须理由充分。只输出JSON。"""


def rerank(brief_dict: dict, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """对召回候选逐个打分排序。candidates 每项需含 name/profile_text。"""
    import json as _json
    brief_str = _json.dumps(brief_dict, ensure_ascii=False)
    scored = []
    for cand in candidates:
        result = chat_json(
            SYSTEM,
            f"投放需求：{brief_str}\n\n候选达人：{cand['profile_text']}",
        )
        scored.append({**cand, **result})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
