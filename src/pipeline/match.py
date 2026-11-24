from src.data.vector_store import build_index, recall
from src.pipeline.brief_parser import parse_brief, to_query_text
from src.pipeline.filters import candidate_ids, hard_filter
from src.pipeline.rerank import rerank


# 能参与匹配的需求字段；count 只作展示，不算"有需求"
_REQUIREMENT_FIELDS = (
    "category", "platform", "audience", "budget_total_yuan",
    "followers_w_range", "style_keywords", "hard_requirements",
)


def _has_requirement(brief_dict: dict) -> bool:
    """判断解析结果里有没有任何可用于匹配的条件。

    全部为 None/空列表时，说明大模型没从原文里读出需求（比如随手打了几个字），
    继续跑召回和重排只会白花模型调用的钱。
    """
    return any(brief_dict.get(k) for k in _REQUIREMENT_FIELDS)


def run_match(brief: str, use_llm_rerank: bool = True, progress=None) -> dict:
    """完整链路：解析 -> 硬过滤 -> 向量召回 -> LLM重排。

    这是项目的业务主入口，界面和评测脚本都调它。

    Args:
        brief: 用户输入的投放需求原文。
        use_llm_rerank: False 时跳过重排，results 里只给占位的
            score=None / reason="未重排"，用于省钱地单独看召回质量。
        progress: 每进入一步回调一次 progress(步骤描述)，由界面渲染成转圈状态；
            传 None 则不报告进度。
    Returns:
        {
          "brief": 输入原文,
          "brief_dict": 解析出的结构化需求,
          "fallback": 硬性条件是否被放宽（True 时界面要提示）,
          "recall_count": 进入重排的候选数,
          "results": Top RERANK_TOP_N 达人，含 daren_id/name/profile_text/score/reason/risk；
            brief 里的 count 只作解析结果展示，不用于截断条数（人工审核需要一批可对比的候选）,
        }
        识别不出需求时提前返回：多一个 "error" 键说明原因，results 为空、其余步骤未执行。
    """
    if progress:
        progress("解析 brief")
    brief_dict = parse_brief(brief)

    # 需求都读不出来就不用往下走了：建索引、召回、重排每步都要花钱
    if not _has_requirement(brief_dict):
        return {
            "brief": brief,
            "brief_dict": brief_dict,
            "error": "没有识别出可用的投放条件（品类/平台/人群/预算/粉丝量均未提及），"
                     "已跳过匹配以免浪费模型调用。请补充需求后再试。",
            "fallback": False,
            "recall_count": 0,
            "results": [],
        }

    if progress:
        progress("加载达人数据与索引")
    df, index = build_index()

    if progress:
        progress("硬性条件过滤")
    rows, fallback = hard_filter(df, brief_dict)

    if progress:
        progress("向量召回")
    ids = recall(to_query_text(brief_dict, brief), df, index)
    ids = candidate_ids(df, rows, ids)
    if not ids:  # 过滤后无候选时退回全量召回
        ids = recall(to_query_text(brief_dict, brief), df, index)
        fallback = True

    candidates = df.loc[ids][["daren_id", "name", "profile_text"]].to_dict("records")

    # 固定给一批候选而不是按 brief 的 count 截断：系统只做初筛，最终由人对比审核
    if use_llm_rerank:
        if progress:
            progress(f"LLM 重排 {len(candidates)} 个候选")
        results = rerank(brief_dict, candidates)
    else:
        results = [{**c, "score": None, "reason": "未重排", "risk": None} for c in candidates]

    return {
        "brief": brief,
        "brief_dict": brief_dict,
        "fallback": fallback,
        "recall_count": len(candidates),
        "results": results,
    }
