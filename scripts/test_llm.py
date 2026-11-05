"""Smoke-test the local DeepSeek-R1-14B GGUF deployment."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from teasys import config
from teasys.llm import get_client, llm_status, warmup


def main():
    print("status:", llm_status())
    t0 = time.perf_counter()
    print("warmup:", warmup())
    print("load seconds:", round(time.perf_counter() - t0, 1))
    client = get_client()
    messages = [
        {"role": "system", "content": "你是茶叶植保专家，回答简洁，使用中文。"},
        {"role": "user", "content": "茶褐枯病的典型症状是什么？用三点回答。"},
    ]
    t0 = time.perf_counter()
    out = client.chat(messages, max_tokens=220)
    dt = time.perf_counter() - t0
    print("--- answer ---")
    print(out.content[:800])
    print("--- stats ---")
    print("latency", round(dt, 1), "s", "chars", len(out.content))
    print("model", out.model)


if __name__ == "__main__":
    main()
