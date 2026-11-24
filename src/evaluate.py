"""评测：召回命中率 + 重排认可率。

评测集 data/eval/briefs.jsonl 每行一个 JSON：
{"brief": "...", "expected_daren_ids": ["d001"], "expected_categories": ["美妆"]}

用法: python -m src.evaluate            # 只跑召回（不调重排LLM，省钱）
      python -m src.evaluate --rerank   # 连重排一起跑
"""
import argparse
import json

from src.config import EVAL_FILE
from src.data.vector_store import build_index, recall
from src.pipeline.brief_parser import parse_brief, to_query_text
from src.pipeline.match import run_match


def load_eval_set() -> list[dict]:
    """读取评测集，每行一个 JSON，字段见模块 docstring。"""
    with open(EVAL_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(rerank: bool):
    """逐条跑评测集，打印命中率。

    Args:
        rerank: True 走 run_match 完整链路，每个候选一次大模型调用，慢且贵；
            False 只做解析 + 向量召回，用来快速判断召回环节是否漏人。
    """
    cases = load_eval_set()
    df, index = build_index()
    hits = 0

    for case in cases:
        if rerank:
            out = run_match(case["brief"])
            top_ids = {str(r["daren_id"]) for r in out["results"]}
            expected = set(case.get("expected_daren_ids") or [])
            hit = bool(top_ids & expected)
            print(f"[rerank] {case['brief'][:30]}... -> Top{len(out['results'])}命中: {hit}")
        else:
            brief_dict = parse_brief(case["brief"])
            ids = recall(to_query_text(brief_dict, case["brief"]), df, index)
            top_ids = {str(df.loc[i]["daren_id"]) for i in ids}
            expected = set(case.get("expected_daren_ids") or [])
            hit = bool(top_ids & expected)
            print(f"[recall] {case['brief'][:30]}... -> Top{len(ids)}命中: {hit}")
        hits += hit

    print(f"\n命中率: {hits}/{len(cases)} = {hits / len(cases):.0%}")
    print("把这条指标和一次bad case修复过程写进一页说明。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="同时评测LLM重排后的命中")
    main(parser.parse_args().rerank)
