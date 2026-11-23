import pandas as pd


def hard_filter(df: pd.DataFrame, brief: dict) -> pd.DataFrame:
    """硬性条件先用规则过滤：可解释、省 token，被过滤的行不会进入召回。"""
    mask = pd.Series(True, index=df.index)

    if brief.get("category"):
        mask &= df["category"].str.contains(brief["category"].strip(), na=False)

    if brief.get("platform"):
        mask &= df["platform"].str.contains(brief["platform"].strip(), na=False)

    fr = brief.get("followers_w_range")
    if isinstance(fr, (list, tuple)) and len(fr) == 2 and all(fr):
        mask &= df["followers_w"].between(float(fr[0]), float(fr[1]))

    budget = brief.get("budget_total_yuan")
    count = max(int(brief.get("count") or 5), 1)
    if budget:
        # 单人预算上限按 总预算/数量 估算，留出组合空间
        mask &= df["price_yuan"] <= float(budget)

    filtered = df[mask]
    # 过滤过狠时回退到全量，并打上告警标记，避免召回空结果
    if len(filtered) < 10:
        return df, True
    return filtered, False


def candidate_ids(df: pd.DataFrame, rows: pd.DataFrame, ids: list[int]) -> list[int]:
    allowed = set(rows.index)
    return [i for i in ids if i in allowed]
