from src.data.vector_store import build_index, recall
from src.pipeline.brief_parser import parse_brief, to_query_text
from src.pipeline.filters import candidate_ids, hard_filter
from src.pipeline.rerank import rerank


def run_match(brief: str, use_llm_rerank: bool = True, progress=None) -> dict:
    """完整链路：解析 -> 硬过滤 -> 向量召回 -> LLM重排。"""
    if progress:
        progress("加载达人数据与索引")
    df, index = build_index()

    if progress:
        progress("解析 brief")
    brief_dict = parse_brief(brief)

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
