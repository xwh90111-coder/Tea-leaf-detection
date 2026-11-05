"""Central configuration for the reproduced tea-disease detection system."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../reproduction
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
REPORT_DIR = DATA_DIR / "reports"
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = Path(os.getenv("TEA_DB_PATH", DATA_DIR / "tea_system.db"))
MODEL_PATH = Path(
    os.getenv(
        "TEA_MODEL_PATH",
        ROOT / "models" / "best.pt",
    )
)
MODEL_FALLBACK = ROOT.parent / "茶叶病害检测" / "tea" / "weights" / "best.pt"

for _d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
IMG_SIZE = int(os.getenv("TEA_IMG_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("TEA_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("TEA_IOU", "0.7"))
MAX_DET = int(os.getenv("TEA_MAX_DET", "300"))
DEVICE = os.getenv("TEA_DEVICE", "0")               # "0" for CUDA, "cpu" otherwise

CLASS_NAMES = ["algal-leaf-spot", "brown-blight", "grey-blight"]
CLASS_NAMES_CN = {
    "algal-leaf-spot": "茶藻斑病",
    "brown-blight": "茶褐枯病",
    "grey-blight": "茶灰枯病",
}
CLASS_COLORS = {
    "algal-leaf-spot": "#2e7d32",
    "brown-blight": "#c62828",
    "grey-blight": "#6d4c41",
}

# ---------------------------------------------------------------------------
# Lightweight LLM (local deployment; data never leaves the machine)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("TEA_LLM_PROVIDER", "llamacpp")    # llamacpp | ollama | openai | none
LLM_BASE_URL = os.getenv("TEA_LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("TEA_LLM_MODEL", "deepseek-r1:14b")
LLM_TIMEOUT = int(os.getenv("TEA_LLM_TIMEOUT", "300"))
LLM_TEMPERATURE = float(os.getenv("TEA_LLM_TEMPERATURE", "0.3"))
LLM_NUM_CTX = int(os.getenv("TEA_LLM_NUM_CTX", "8192"))
LLM_MAX_TOKENS = int(os.getenv("TEA_LLM_MAX_TOKENS", "1300"))
LLM_THREADS = int(os.getenv("TEA_LLM_THREADS", "20"))
LLM_GGUF_PATH = Path(
    os.getenv(
        "TEA_LLM_GGUF",
        ROOT / "models" / "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    )
)

# ---------------------------------------------------------------------------
# Retrieval (D2 knowledge base)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("TEA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_SIZE = int(os.getenv("TEA_CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("TEA_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TEA_TOP_K", "4"))

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
TERMINAL_ID = os.getenv("TEA_TERMINAL_ID", "CAM-A2A5328-01")
APP_HOST = os.getenv("TEA_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("TEA_PORT", "8050"))
MAX_VIDEO_SECONDS = int(os.getenv("TEA_MAX_VIDEO_SECONDS", "60"))
MODEL_VERSION = os.getenv("TEA_MODEL_VERSION", "yolo12n-tea-400e")


def resolve_model_path() -> Path:
    """Return the reproduced checkpoint, falling back to the original run."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    if MODEL_FALLBACK.exists():
        return MODEL_FALLBACK
    raise FileNotFoundError(
        f"No model checkpoint found at {MODEL_PATH} or {MODEL_FALLBACK}"
    )
