import json
import os
import re

import numpy as np
from dotenv import load_dotenv

from src.config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL, LLM_API, LLM_MODEL

load_dotenv()


def _client():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请复制 .env.example 为 .env 并填入")
    import dashscope
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    return dashscope


def _extract_json(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = min(i for i in (text.find("{"), text.find("[")) if i != -1)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start : end + 1])


def _content_text(resp) -> str:
    content = resp.output.choices[0].message.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def chat_json(system: str, user: str, temperature: float = 0.2):
    dashscope = _client()
    if LLM_API == "multimodal":
        call = dashscope.MultiModalConversation.call
        messages = [
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": [{"text": user}]},
        ]
    else:
        call = dashscope.Generation.call
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    resp = call(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        result_format="message",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"通义调用失败: {resp.code} {resp.message}")
    return _extract_json(_content_text(resp))


def embed(texts: list[str]) -> np.ndarray:
    dashscope = _client()
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = dashscope.TextEmbedding.call(
            model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH], dimension=EMBED_DIM
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding 调用失败: {resp.code} {resp.message}")
        vectors.extend(item["embedding"] for item in resp.output["embeddings"])
    arr = np.array(vectors, dtype="float32")
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
