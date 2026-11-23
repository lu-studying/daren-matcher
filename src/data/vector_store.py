import hashlib
import json

import faiss
import numpy as np

from src.config import CACHE_DIR, EMBED_DIM, EMBED_MODEL, RECALL_TOP_K
from src.data.ingest import load_influencers
from src.llm import embed


def _cache_key(df) -> str:
    # 模型名和维度必须进 key：换 EMBED_MODEL/EMBED_DIM 后旧向量不可复用
    payload = [EMBED_MODEL, EMBED_DIM, df["profile_text"].tolist()]
    return hashlib.md5(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _read_index(path):
    # faiss 的 C++ 文件 IO 在 Windows 上处理不了中文路径，经内存缓冲中转
    data = path.read_bytes()
    pos = 0

    def read_chunk(size: int) -> bytes:
        nonlocal pos
        chunk = data[pos : pos + size]
        pos += size
        return chunk

    return faiss.read_index(faiss.PyCallbackIOReader(read_chunk))


def build_index(force: bool = False):
    """把达人画像向量化并缓存为 FAISS 索引。数据没变时直接复用缓存。"""
    df = load_influencers()
    key = _cache_key(df)
    vec_path = CACHE_DIR / f"{key}.npy"
    idx_path = CACHE_DIR / f"{key}.faiss"
    if (vec_path.exists() and idx_path.exists() and idx_path.stat().st_size > 0
            and not force):
        return df, _read_index(idx_path)

    vectors = embed(df["profile_text"].tolist())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    idx_path.write_bytes(faiss.vector_to_array(writer.data).tobytes())
    return df, index


def recall(query_text: str, df, index, top_k: int = RECALL_TOP_K) -> list[int]:
    """语义召回，返回 df 的行号列表（按相似度降序）。"""
    q = embed([query_text])
    _, ids = index.search(q, min(top_k, len(df)))
    return [int(i) for i in ids[0] if i >= 0]
