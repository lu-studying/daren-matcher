import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dashscope

from src.config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

MODELS = [EMBED_MODEL]
BATCH = ["美妆博主真实测评", "数码开箱达人"] * (EMBED_BATCH // 2)

for m in MODELS:
    print(f"model = {m}")

    r = dashscope.TextEmbedding.call(model=m, input=BATCH, dimension=EMBED_DIM)
    if r.status_code == 200:
        vecs = r.output["embeddings"]
        print(f"    [200 OK] dimension={EMBED_DIM} 可用，本批 {len(vecs)} 条")
    else:
        print(f"    [{r.status_code}] code    = {r.code}")
        print(f"                    message = {r.message}")

    r2 = dashscope.TextEmbedding.call(model=m, input=["默认维度测试"])
    if r2.status_code == 200:
        print(f"    不传 dimension 时的原生维度 = {len(r2.output['embeddings'][0]['embedding'])}")
    print("-" * 70)
