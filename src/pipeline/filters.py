import pandas as pd


def hard_filter(df: pd.DataFrame, brief: dict) -> pd.DataFrame:
    """硬性条件先用规则过滤：可解释、省 token，被过滤的行不会进入召回。

    逐条 AND 起来：品类、平台、粉丝量级区间、单人报价。brief 里没提的字段
    （解析结果为 None）直接跳过，不做约束。

    Args:
        df: load_influencers() 返回的达人表。
        brief: parse_brief() 的解析结果。
    Returns:
        (rows, fallback) 二元组。
        rows：保留下来的 DataFrame，索引仍是原行号，candidate_ids 靠这个取交集；
        fallback：True 表示过滤后不足 10 条、已退回全量表，界面需提示"条件已放宽"。
    """
    mask = pd.Series(True, index=df.index)

    if brief.get("category"):
        mask &= df["category"].str.contains(brief["category"].strip(), na=False)

    if brief.get("platform"):
        mask &= df["platform"].str.contains(brief["platform"].strip(), na=False)

    fr = brief.get("followers_w_range")
    if isinstance(fr, (list, tuple)) and len(fr) == 2 and all(fr):
        mask &= df["followers_w"].between(float(fr[0]), float(fr[1]))

    budget = brief.get("budget_total_yuan")
    if budget:
        # 只判断"单个达人请不请得起"，不按人数分摊：预算本来就不是平均分配的
        mask &= df["price_yuan"] <= float(budget)

    filtered = df[mask]
    # 过滤过狠时回退到全量，并打上告警标记，避免召回空结果
    if len(filtered) < 10:
        return df, True
    return filtered, False


def candidate_ids(df: pd.DataFrame, rows: pd.DataFrame, ids: list[int]) -> list[int]:
    """把语义召回的行号与硬过滤保留的行号取交集。

    Args:
        rows: hard_filter() 的结果，这里只用到它的 index。
        ids: recall() 返回的行号，已按相似度降序。
    Returns:
        同时满足"语义相似"和"硬性条件"的行号，保持原来的相似度顺序。
        返回空列表时由 run_match 决定是否退回全量召回。
    """
    allowed = set(rows.index)
    return [i for i in ids if i in allowed]
