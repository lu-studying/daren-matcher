import pandas as pd

# 方向条件：决定"找哪类达人"，候选不足时不放宽
_CORE_KEYS = ("category", "platform")
# 量级条件：粉丝量与单人报价，候选不足时先放宽这两项
_SCALE_KEYS = ("followers_w_range", "budget_total_yuan")
# 放宽触发线：低于这个数就给一批可对比的候选；按数据量的一成算，但不少于 3 条
_MIN_KEEP_RATIO = 0.1
_MIN_KEEP_FLOOR = 3


def _build_mask(df: pd.DataFrame, brief: dict, keys: tuple) -> pd.Series:
    """按 keys 指定的字段拼布尔掩码，brief 里没提的字段不参与约束。

    Args:
        df: load_influencers() 返回的达人表。
        brief: parse_brief() 的解析结果。
        keys: 要参与约束的字段名，只会传 _CORE_KEYS 或 _SCALE_KEYS。
    Returns:
        与 df 等长的布尔 Series，True 表示该行通过全部指定条件。
    """
    mask = pd.Series(True, index=df.index)

    if "category" in keys and brief.get("category"):
        mask &= df["category"].str.contains(brief["category"].strip(), na=False)

    if "platform" in keys and brief.get("platform"):
        mask &= df["platform"].str.contains(brief["platform"].strip(), na=False)

    if "followers_w_range" in keys:
        fr = brief.get("followers_w_range")
        if isinstance(fr, (list, tuple)) and len(fr) == 2:
            # 单侧为 null 表示该侧不限（如「10万以上」是 [10, null]），只约束有值的一侧
            low, high = fr
            if low is not None:
                mask &= df["followers_w"] >= float(low)
            if high is not None:
                mask &= df["followers_w"] <= float(high)

    if "budget_total_yuan" in keys:
        budget = brief.get("budget_total_yuan")
        if budget:
            # 只判断"单个达人请不请得起"，不按人数分摊：预算本来就不是平均分配的
            mask &= df["price_yuan"] <= float(budget)

    return mask


def hard_filter(df: pd.DataFrame, brief: dict) -> tuple:
    """硬性条件先用规则过滤：可解释、省 token，被过滤的行不会进入召回。

    条件分两级：方向条件（品类/平台）和量级条件（粉丝量级/单人报价）。
    先要求两级同时满足；命中不足时只放宽量级条件，仍无命中才放宽方向条件。
    任何时候都不退回全表——全量候选会让重排的模型调用数失控。

    Args:
        df: load_influencers() 返回的达人表。
        brief: parse_brief() 的解析结果。
    Returns:
        (rows, relaxed_reason) 二元组。
        rows：保留下来的 DataFrame，索引仍是原行号，candidate_ids 靠这个取交集；
        relaxed_reason：None 表示条件原样生效；str 说明放宽了哪一级，界面据此提示。
        两级条件都没有命中时返回空表，由上层按"无匹配"处理，不再绕过硬过滤。
    """
    core = _build_mask(df, brief, _CORE_KEYS)
    scale = _build_mask(df, brief, _SCALE_KEYS)
    strict = df[core & scale]
    min_keep = max(_MIN_KEEP_FLOOR, int(len(df) * _MIN_KEEP_RATIO))

    # 没提品类/平台时 core 全为 True，放宽它等于不加任何约束，没有意义
    if len(strict) >= min_keep or core.all():
        return strict, None

    if core.any():
        return df[core], f"满足全部条件的只有 {len(strict)} 位，已放宽粉丝量与预算"
    # scale 全为 True 说明 brief 根本没给量级条件，这时"按量级筛选"等于退回全表
    if not scale.all() and scale.any():
        return df[scale], "品类/平台无命中，已放宽方向条件，仅按粉丝量与预算筛选"
    return strict, None


def candidate_ids(df: pd.DataFrame, rows: pd.DataFrame, ids: list) -> list:
    """把语义召回的行号与硬过滤保留的行号取交集。

    Args:
        rows: hard_filter() 的结果，这里只用到它的 index。
        ids: recall() 返回的行号，已按相似度降序。
    Returns:
        同时满足"语义相似"和"硬性条件"的行号，保持原来的相似度顺序。
        返回空列表时由 run_match 决定用过滤结果本身补位。
    """
    allowed = set(rows.index)
    return [i for i in ids if i in allowed]