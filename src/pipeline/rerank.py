import re

from src.config import RERANK_TOP_N
from src.llm import chat_text, try_extract_json

SYSTEM = """你是资深达人投放顾问。根据投放需求评估一个达人候选，输出JSON：
{"score": 0-100整数, "reason": "为什么匹配，一句话，引用具体数据", "risk": "最主要的一个风险点，没有则填 null"}
评分依据：品类契合 > 人群与风格契合 > 量级与预算合理性。宁低勿高，80分以上必须理由充分。
reason 内不要使用英文双引号，需要强调时用「」；score 只写纯数字，不要带「分」。只输出JSON。"""

_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"(.*?)"\s*,\s*"risk"', re.S)
_RISK_RE = re.compile(r'"risk"\s*:\s*"?(.*?)"?\s*[,}]', re.S)


def _to_score(value) -> int:
    """把模型给的分数转成 0-100 的整数。

    兼容 85、"85"、"85分" 等写法；识别不出时返回 0，与 prompt 的「宁低勿高」一致。
    """
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return 0
    return max(0, min(100, int(round(float(match.group())))))


def _regex_fields(text: str) -> dict:
    """整段 JSON 解析失败时，按字段名逐个正则抽取 score/reason/risk。

    模型偶发输出未转义引号或「85分」，完整 JSON 已不可救；
    下游只用这三个字段，按字段名抠出来比整段重试更省一次调用。
    """
    score_m = _SCORE_RE.search(text)
    reason_m = _REASON_RE.search(text)
    risk_m = _RISK_RE.search(text)
    risk = risk_m.group(1).strip() if risk_m else ""
    return {
        "score": _to_score(score_m.group(1) if score_m else None),
        "reason": reason_m.group(1).strip() if reason_m else "模型输出解析失败",
        "risk": None if risk in ("", "null", "None") else risk,
    }


def parse_score_reply(text: str) -> dict:
    """把打分回复规整成 {score, reason, risk} 三个字段。

    先按完整 JSON 解析；不合法则退化为正则抽取。任何输入都不抛错，
    保证一个候选的坏回复不会让整轮重排中断。
    """
    data = try_extract_json(text)
    if isinstance(data, dict):
        return {
            "score": _to_score(data.get("score")),
            "reason": str(data.get("reason") or "").strip(),
            "risk": data.get("risk") or None,
        }
    return _regex_fields(text)


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
        单条回复解析失败时该项 score 记 0 并标注原因，不影响其余候选。
    Raises:
        RuntimeError: 接口调用本身失败（缺 key、非 200），这类问题要立刻暴露而不是降级。
    """
    import json as _json
    brief_str = _json.dumps(brief_dict, ensure_ascii=False)
    scored = []
    for cand in candidates:
        text = chat_text(
            SYSTEM,
            f"投放需求：{brief_str}\n\n候选达人：{cand['profile_text']}",
        )
        scored.append({**cand, **parse_score_reply(text)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]