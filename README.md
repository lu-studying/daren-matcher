# 达人智能匹配工具（daren-matcher）

输入自然语言投放 brief，自动完成：**Brief 解析 → 硬性条件过滤 → Embedding 语义召回 → 通义千问重排**，
输出带匹配理由与风险提示的达人推荐清单，并记录真实使用反馈。

## 目录结构

```
app/streamlit_app.py      Streamlit 前端（部署入口）
src/
  config.py               模型名、路径、召回/重排参数
  llm.py                  通义（DashScope）封装：chat_json + embed
  data/
    ingest.py             达人 CSV 加载、画像文本拼接
    vector_store.py       FAISS 索引构建（带缓存）与召回
  pipeline/
    brief_parser.py       brief -> 结构化需求 JSON
    filters.py            预算/量级/品类/平台硬过滤（可解释、省token）
    rerank.py             LLM 逐候选打分：score/reason/risk
    match.py              全链路编排
  evaluate.py             评测：召回命中率 / 重排命中
  feedback.py             采纳/不合适事件落库（sqlite）
data/
  raw/influencers.csv     达人库（当前为12条示例，替换为真实整理数据）
  eval/briefs.jsonl       评测集
  cache/                  向量缓存（自动生成，勿提交）
docs/one-pager.md         一页说明模板
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 DASHSCOPE_API_KEY（阿里云百炼控制台获取）

# 启动应用
streamlit run app/streamlit_app.py

# 跑评测（先召回，省钱；加 --rerank 连重排一起评）
python -m src.evaluate
```

## 匹配链路设计要点

- **硬条件走规则、软匹配走模型**：预算/量级/品类先过滤，LLM 只评估真正值得看的候选。
- **过滤过狠自动回退**：候选少于 10 时放宽并明确标注，避免空结果。
- **画像文本即召回质量**：粉丝量、互动率、报价都写进 embedding 文本，改数据描述就是改召回。
- **每次使用都留痕**：search/adopt/reject 事件写入 sqlite，页脚统计即「真实使用数据」。

## 下一步（对应两周计划）

1. 把 `data/raw/influencers.csv` 扩充到 200+ 真实整理达人
2. 评测集扩到 10-15 条，记录一次 bad case 修复过程
3. 部署到 HuggingFace Spaces / Streamlit Cloud
4. 邀请 3-5 位目标用户试用，回填一页说明
