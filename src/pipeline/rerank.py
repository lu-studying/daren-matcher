from src.config import RERANK_TOP_N
from src.llm import chat_json

SYSTEM = """你是资深达人投放顾问。根据投放需求评估一个达人候选，输出JSON：
{"score": 0-100整数, "reason": "为什么匹配，一句话，引用具体数据", "risk": "最主要的一个风险点，没有则填 null"}
评分依据：品类契合 > 人群与风格契合 > 量级与预算合理性。宁低勿高，80分以上必须理由充分。只输出JSON。"""


def rerank(brief_dict: dict, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """对召回候选逐个打分排序。candidates 每项需含 name/profile_text。

    每个候选单独发一次请求，N 个候选就是 N 次大模型调用，是全链路最贵的一步；
    所以前面要先做硬过滤和向量召回，把候选压到几十个以内。

    Args:
        brief_dict: parse_brief() 的结果，序列化后作为评分依据。
        candidates: recall 命中行的 [{daren_id, name, profile_text}, ...]。
        top_n: 返回条数上限。
    Returns:
        按 score 降序的前 top_n 项，每项在原有字段上合并 score / reason / risk。
    """
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
