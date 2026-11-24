import json
import os
import re

import numpy as np
from dotenv import load_dotenv

from src.config import EMBED_BATCH, EMBED_DIM, EMBED_MODEL, LLM_API, LLM_MODEL

load_dotenv()


def _client():
    """返回已设置好 api_key 的 dashscope 模块。

    Key 只从环境变量读：本地由 load_dotenv() 注入 .env，云端由 Streamlit Secrets 注入。
    缺失时立刻抛错，避免带着空密钥发请求后只拿到一个难以定位的 401。
    """
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请复制 .env.example 为 .env 并填入")
    import dashscope
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    return dashscope


def _extract_json(text: str):
    """从模型回复中抠出并解析 JSON。

    容忍两种常见的不干净输出：被 ```json 代码块包裹、JSON 前后夹带解释文字。
    定位方式是取第一个 { 或 [ 到最后一个 } 或 ]，因此要求模型只输出一段 JSON。

    Args:
        text: 模型回复的纯文本。
    Returns:
        解析后的 dict 或 list，取决于模型给的是对象还是数组。
    Raises:
        json.JSONDecodeError: 回复里找不到合法 JSON。异常信息里附带被解析的原文，
            否则只报"第 77 列非法"无法判断是未转义引号还是「85分」这类带单位的数值。
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise json.JSONDecodeError(f"回复中没有 JSON（原文：{text!r}）", text, 0)
    end = max(text.rfind("}"), text.rfind("]"))
    payload = text[min(starts) : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"{e.msg}（原文：{payload!r}）", e.doc, e.pos) from e


def try_extract_json(text: str):
    """同 _extract_json，但解析失败时返回 None 而不抛错。

    供需要自己兜底的调用方使用（如重排时单个候选回复不合法，宁可降级也不能中断整轮）。
    """
    try:
        return _extract_json(text)
    except json.JSONDecodeError:
        return None


def _content_text(resp) -> str:
    """把响应里的 message.content 拍平成字符串。

    Generation 端点返回 str，MultiModalConversation 端点返回 [{"text": ...}]；
    这里统一处理，两个端点的结果才能共用同一条解析路径。
    """
    content = resp.output.choices[0].message.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def chat_text(system: str, user: str, temperature: float = 0.2) -> str:
    """发一轮 system + user 对话，返回模型回复的原始文本。

    按 config.LLM_API 在两个百炼端点间切换，消息体格式不同：
      "generation" -> Generation.call，content 是纯字符串
      "multimodal" -> MultiModalConversation.call，content 是 [{"text": ...}]
    LLM_API 必须与 LLM_MODEL 实际提供的端点配对，配错返回 400 InvalidParameter: url error。

    Args:
        system: 角色设定与输出格式约束。
        user: 本轮要处理的实际内容。
        temperature: 采样温度。解析和打分都要求可复现，故默认压低到 0.2。
    Returns:
        回复文本；调用方若需要自行容错解析，就用这个函数而不是 chat_json。
    Raises:
        RuntimeError: 状态码非 200，附带百炼的 code 与 message。
    """
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
    return _content_text(resp)


def chat_json(system: str, user: str, temperature: float = 0.2):
    """发一轮对话并把回复解析成 JSON，参数与 chat_text 相同。

    Returns:
        dict 或 list，由 _extract_json 从回复中解析得到。
    Raises:
        json.JSONDecodeError: 回复不是合法 JSON，异常信息里附带原文。
    """
    return _extract_json(chat_text(system, user, temperature))


def embed(texts: list[str]) -> np.ndarray:
    """把一批文本转成单位归一化的向量矩阵。

    Args:
        texts: 待向量化的文本，按 config.EMBED_BATCH 分批请求（单次请求有条数上限）。
    Returns:
        shape=(len(texts), EMBED_DIM) 的 float32 矩阵，每行模长为 1。
        归一化是 FAISS 用 IndexFlatIP 的前提：内积等于向量夹角余弦，
        所以 index.search 返回的距离可以直接当相似度用，不用再换 IndexFlatL2。
    Raises:
        RuntimeError: 状态码非 200，附带百炼的 code 与 message。
    """
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
