import hashlib
import json

import faiss
import numpy as np

from src.config import CACHE_DIR, EMBED_DIM, EMBED_MODEL, RECALL_TOP_K
from src.data.ingest import load_influencers
from src.llm import embed


def _cache_key(df) -> str:
    """达人画像内容 + embedding 模型名 + 维度 的哈希，作为缓存文件名。"""
    # 模型名和维度必须进 key：换 EMBED_MODEL/EMBED_DIM 后旧向量不可复用
    payload = [EMBED_MODEL, EMBED_DIM, df["profile_text"].tolist()]
    return hashlib.md5(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _read_index(path):
    """从磁盘字节还原 FAISS 索引对象。

    faiss 的 C++ 文件 IO 在 Windows 上打不开中文路径，因此先用 Python 读成 bytes，
    再通过 PyCallbackIOReader 回调从内存缓冲喂给 faiss。写入侧同理见 build_index。
    """
    data = path.read_bytes()
    pos = 0

    def read_chunk(size: int) -> bytes:
        nonlocal pos
        chunk = data[pos : pos + size]
        pos += size
        return chunk

    return faiss.read_index(faiss.PyCallbackIOReader(read_chunk))


def build_index(force: bool = False):
    """把达人画像向量化并缓存为 FAISS 索引。数据没变时直接复用缓存。

    向量化要按批次调用 embedding 接口，是链路里最慢的一步，所以用
    {key}.npy / {key}.faiss 两个文件缓存，key 由 _cache_key 算出。

    Args:
        force: True 时忽略已有缓存，重新向量化并覆盖写盘。
    Returns:
        (df, index)：load_influencers() 的表和建好的索引，
        index 里第 i 条向量对应 df 的第 i 行，行号可互相还原。
    """
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
    """语义召回，返回 df 的行号列表（按相似度降序）。

    Args:
        query_text: 由 to_query_text() 拼出来的查询文本，内部会再调一次 embed。
        top_k: 期望条数，实际取 min(top_k, 达人数)。
    Returns:
        命中行号；不足 top_k 条时 faiss 用 -1 填充，已在这里过滤掉。
    """
    q = embed([query_text])
    _, ids = index.search(q, min(top_k, len(df)))
    return [int(i) for i in ids[0] if i >= 0]
