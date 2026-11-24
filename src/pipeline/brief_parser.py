from src.llm import chat_json

SYSTEM = """你是营销投放领域的brief分析专家。把用户的自然语言投放需求解析成JSON，字段：
{
  "category": "品类，如 美妆/食品饮料/3C数码，无法判断填 null",
  "platform": "目标平台，如 抖音/小红书/B站，未指定填 null",
  "audience": "目标人群描述，未提及填 null",
  "budget_total_yuan": "总预算数字（人民币），未提及填 null",
  "followers_w_range": [下限万, 上限万]，不知道的一侧填 null，未提及整项填 null,
  "count": "达人数量，未提及填 null",
  "style_keywords": ["内容风格关键词，2-5个"],
  "hard_requirements": ["其他硬性约束，如 必须原创视频/排除擦边内容"]
}
followers_w_range 必须按原意分到下限/上限两侧：「10万以上」「10万+」→ [10, null]；「1-20万」→ [1, 20]；
「10万以内」→ [null, 10]；「不限」「没有要求」→ null。不要把单侧条件塞进另一侧。
count 只填用户明确说的数量：「一个」「一位」→ 1，「5个」→ 5；没说数量就填 null，不要套默认值。
只输出JSON。信息不足时填 null，不要编造。"""


def parse_brief(brief: str) -> dict:
    """用大模型把自然语言投放需求解析成结构化字典。

    Returns:
        字段由 SYSTEM 约定：category / platform / audience / budget_total_yuan /
        followers_w_range / count / style_keywords / hard_requirements，
        信息不足时为 None。下游按 None 跳过对应过滤，因此不会误杀候选；
        count 只作需求信息展示，不用于截断推荐条数（系统固定给一批候选供人工审核）。
    """
    return chat_json(SYSTEM, brief)


def to_query_text(brief_dict: dict, raw_brief: str) -> str:
    """把解析结果拼成召回用的查询文本。"""
    parts = [raw_brief]
    for key in ("category", "platform", "audience"):
        if brief_dict.get(key):
            parts.append(str(brief_dict[key]))
    parts.extend(brief_dict.get("style_keywords") or [])
    return "。".join(str(p) for p in parts)
