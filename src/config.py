from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LLM_MODEL = "deepseek-v4.1-flash-1017"          # 解析/重排用，便宜且够用；追求质量可换 qwen-max
# LLM_API 必须与 LLM_MODEL 实际提供的端点配对，配错百炼返回 400 InvalidParameter: url error
# 端点归属无法从模型名推断，换模型时两个值各试一次。已实测通过的组合：
#   "generation" + qwen-plus
#   "multimodal" + deepseek-v4.1-flash-1017（同一模型名走 generation 会报 url error）
LLM_API = "multimodal"
EMBED_MODEL = "qwen3.7-text-embedding-flash"
EMBED_DIM = 1024
EMBED_BATCH = 10                  # 单次请求的文本条数上限，换 EMBED_MODEL 时需按该模型的限制核对

INFLUENCER_FILE = ROOT / "data" / "raw" / "influencers.csv"
EVAL_FILE = ROOT / "data" / "eval" / "briefs.jsonl"
CACHE_DIR = ROOT / "data" / "cache"
FEEDBACK_DB = ROOT / "data" / "feedback.db"

RECALL_TOP_K = 50
RERANK_TOP_N = 10
