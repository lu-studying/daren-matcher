from src.llm import chat_json

SYSTEM = """你是营销投放领域的brief分析专家。把用户的自然语言投放需求解析成JSON，字段：
{
  "category": "品类，如 美妆/食品饮料/3C数码，无法判断填 null",
  "platform": "目标平台，如 抖音/小红书/B站，未指定填 null",
  "audience": "目标人群描述",
  "budget_total_yuan": "总预算数字（人民币），未提及填 null",
  "followers_w_range": [粉丝量级下限万, 上限万]，未提及填 null,
  "count": "期望达人数量，默认5",
  "style_keywords": ["内容风格关键词，2-5个"],
  "hard_requirements": ["其他硬性约束，如 必须原创视频/排除擦边内容"]
}
只输出JSON。信息不足时填 null，不要编造。"""


def parse_brief(brief: str) -> dict:
    return chat_json(SYSTEM, brief)


def to_query_text(brief_dict: dict, raw_brief: str) -> str:
    """把解析结果拼成召回用的查询文本。"""
    parts = [raw_brief]
    for key in ("category", "platform", "audience"):
        if brief_dict.get(key):
            parts.append(str(brief_dict[key]))
    parts.extend(brief_dict.get("style_keywords") or [])
    return "。".join(str(p) for p in parts)
