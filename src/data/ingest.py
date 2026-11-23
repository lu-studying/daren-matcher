import pandas as pd

from src.config import INFLUENCER_FILE

REQUIRED_COLS = [
    "daren_id", "name", "platform", "category", "followers_w",
    "engagement_rate", "avg_views", "price_yuan", "content_summary", "tags",
]


def load_influencers(path=INFLUENCER_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"达人数据缺少字段: {missing}")
    df["profile_text"] = df.apply(profile_text, axis=1)
    return df


def profile_text(row: pd.Series) -> str:
    return (
        f"{row['name']}，{row['platform']}平台{row['category']}类达人，"
        f"粉丝{row['followers_w']}万，近30天互动率{row['engagement_rate']}%，"
        f"场均播放约{row['avg_views']}，单条报价约{row['price_yuan']}元。"
        f"近期内容：{row['content_summary']}。风格标签：{row['tags']}。"
    )
