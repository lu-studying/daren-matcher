import pandas as pd

from src.config import INFLUENCER_FILE

REQUIRED_COLS = [
    "daren_id", "name", "platform", "category", "followers_w",
    "engagement_rate", "avg_views", "price_yuan", "content_summary", "tags",
]


def load_influencers(path=INFLUENCER_FILE) -> pd.DataFrame:
    """读取达人 CSV，校验字段完整，并附加 profile_text 列。

    Args:
        path: CSV 路径，默认 config.INFLUENCER_FILE。
    Returns:
        DataFrame，索引为 0..n-1 的连续整数。整条链路都用这个行号定位达人
        （召回返回行号、重排按行号取数据），所以这里不能排序或重置索引。
        profile_text 列是向量化和 LLM 重排共用的画像描述。
    Raises:
        ValueError: 缺少 REQUIRED_COLS 中声明的字段。
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"达人数据缺少字段: {missing}")
    df["profile_text"] = df.apply(profile_text, axis=1)
    return df


def profile_text(row: pd.Series) -> str:
    """把一行达人的结构化字段拼成一句自然语言画像。

    这段文本同时喂给 embedding 和 LLM 重排，所以字段顺序和措辞一变，
    向量缓存 key 就变，全量重建索引。
    """
    return (
        f"{row['name']}，{row['platform']}平台{row['category']}类达人，"
        f"粉丝{row['followers_w']}万，近30天互动率{row['engagement_rate']}%，"
        f"场均播放约{row['avg_views']}，单条报价约{row['price_yuan']}元。"
        f"近期内容：{row['content_summary']}。风格标签：{row['tags']}。"
    )
