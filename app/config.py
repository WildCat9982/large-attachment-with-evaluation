"""All tunables in one place; every value can be overridden by an env var."""
import os

DATA_DIR = os.getenv("DATA_DIR", "/data")
SQLITE_PATH = os.path.join(DATA_DIR, "app.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
TABLE_DIR = os.path.join(DATA_DIR, "tables")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

MAX_UPLOAD_MB = 50  # keep in sync with .streamlit/config.toml
ALLOWED_EXT = ["pdf", "docx", "txt", "html", "htm", "csv", "json", "xlsx"]

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")  # answers, final summary
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "claude-haiku-4-5-20251001")  # bulk steps

# Retrieval
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
TOP_K = int(os.getenv("TOP_K", "6"))

# Summarisation: the document is read in windows, each summarised, then merged.
SUMMARY_WINDOW_CHARS = int(os.getenv("SUMMARY_WINDOW_CHARS", "40000"))
# Cost guard. Above this many windows, an evenly spaced sample is read instead
# and the UI says so. 0 = no cap (a 50MB text file is ~1,300 windows).
SUMMARY_MAX_WINDOWS = int(os.getenv("SUMMARY_MAX_WINDOWS", "150"))
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "4"))

# Conversation context kept in Redis
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
CHAT_TTL_SECONDS = int(os.getenv("CHAT_TTL_SECONDS", str(24 * 3600)))

# Text-to-SQL guard rails for tabular files
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
SQL_TIMEOUT_SECONDS = int(os.getenv("SQL_TIMEOUT_SECONDS", "15"))

# Usage tracking and evaluation
# Prices in USD per million tokens, used to turn logged tokens into cost.
def _price(name):
    return float(os.getenv(name) or 0)


PRICES = {
    LLM_FAST_MODEL: (_price("LLM_FAST_PRICE_IN"), _price("LLM_FAST_PRICE_OUT")),
    LLM_MODEL: (_price("LLM_PRICE_IN"), _price("LLM_PRICE_OUT")),
}
PRICES_SET = any(p > 0 for pair in PRICES.values() for p in pair)

# The model that scores answers during evaluation. Ideally not the model being scored.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or LLM_MODEL
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "3"))
EVAL_DIR = os.getenv("EVAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
