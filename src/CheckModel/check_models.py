import os

import dashscope

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

# 从控制台「模型 Code」那一列原样复制你要试的模型
MODELS = ["qwen3.8-flash", "deepseek-v4.1-flash"]

TEXT = "你好，用一句话回答我"


def report(label, resp):
    if resp.status_code == 200:
        print(f"    [200 OK]   {label}")
        return True
    print(f"    [{resp.status_code}]  {label}")
    print(f"           code    = {resp.code}")
    print(f"           message = {resp.message}")
    return False


def text_of(resp):
    content = resp.output["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


for m in MODELS:
    print(f"model = {m}")

    gen = dashscope.Generation.call(
        model=m, messages=[{"role": "user", "content": TEXT}], result_format="message"
    )
    if report("Generation.call（纯文本端点）", gen):
        print(f"           -> {text_of(gen)[:40]!r}")

    mm = dashscope.MultiModalConversation.call(
        model=m, messages=[{"role": "user", "content": [{"text": TEXT}]}]
    )
    if report("MultiModalConversation.call（多模态端点）", mm):
        print(f"           -> {text_of(mm)[:40]!r}")

    print("-" * 70)
