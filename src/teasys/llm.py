"""Local large-language-model client.

Default deployment: DeepSeek-R1-Distill-Qwen-14B (Q4_K_M GGUF) through
llama.cpp, running fully offline.  Ollama and OpenAI-compatible endpoints are
supported as drop-in alternatives via environment variables.

The paper deploys DeepSeek-R1-14B locally so that tea-garden data never leaves
the machine; this module keeps that property and exposes a health check used by
the UI.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from . import config


@dataclass
class LLMResult:
    content: str
    model: str
    latency_s: float
    provider: str
    fallback: bool = False


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek-R1 style reasoning blocks from the final answer."""
    if not text:
        return text
    cleaned = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.S | re.I)
    if re.search(r"<think[^>]*>", cleaned, flags=re.I):
        # unterminated reasoning block: drop everything from the tag onwards
        cleaned = re.split(r"<think[^>]*>", cleaned, maxsplit=1, flags=re.I)[0]
    return cleaned.strip()


_LLAMA = None
_LLAMA_LOCK = threading.Lock()
_LLAMA_INFER = threading.Lock()


def _llama_model():
    """Lazy-load the GGUF model once (thread-safe, shared across callbacks)."""
    global _LLAMA
    if _LLAMA is None:
        with _LLAMA_LOCK:
            if _LLAMA is None:
                import llama_cpp
                from llama_cpp import Llama

                # Disable the CPU weight-repacking buffer (adds ~6.4 GB RAM for a
                # 14B Q4 model and is not needed for correct inference).
                _orig = llama_cpp.llama_model_default_params

                def _params():
                    p = _orig()
                    try:
                        p.use_extra_bufts = False
                    except Exception:
                        pass
                    return p

                llama_cpp.llama_model_default_params = _params

                path = Path(config.LLM_GGUF_PATH)
                if not path.exists():
                    raise FileNotFoundError(f"GGUF model not found: {path}")
                _LLAMA = Llama(
                    model_path=str(path),
                    n_ctx=config.LLM_NUM_CTX,
                    n_threads=config.LLM_THREADS,
                    n_batch=512,
                    n_ubatch=512,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
    return _LLAMA


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    # -- health -----------------------------------------------------------
    def health(self) -> dict[str, Any]:
        if self.provider == "none":
            return {"available": False, "model": self.model, "provider": "none", "detail": "LLM 已禁用"}
        if self.provider == "llamacpp":
            path = Path(config.LLM_GGUF_PATH)
            return {
                "available": path.exists(),
                "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
                "provider": "llamacpp",
                "detail": (
                    "已加载（本地 GGUF）" if _LLAMA is not None else
                    ("模型文件就绪，首次调用时加载" if path.exists() else "未找到 GGUF 模型文件")
                ),
                "path": str(path),
                "size_gb": round(path.stat().st_size / 1024 ** 3, 2) if path.exists() else 0,
            }
        try:
            if self.provider == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=3)
                r.raise_for_status()
                models = [m.get("name", "") for m in r.json().get("models", [])]
                available = any(self.model.split(":")[0] in m for m in models)
                return {
                    "available": available,
                    "model": self.model,
                    "provider": "ollama",
                    "models": models,
                    "detail": "已就绪" if available else "服务在线，但模型未下载",
                }
            r = requests.get(f"{self.base_url}/v1/models", timeout=3)
            r.raise_for_status()
            models = [m.get("id", "") for m in r.json().get("data", [])]
            return {
                "available": any(self.model == m or self.model in m for m in models),
                "model": self.model,
                "provider": "openai",
                "models": models,
                "detail": "已就绪",
            }
        except Exception as e:
            return {
                "available": False,
                "model": self.model,
                "provider": self.provider,
                "detail": "未连接：请先启动本地模型服务",
                "error": str(e)[:120],
            }

    # -- chat -------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResult:
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        num_ctx = num_ctx or config.LLM_NUM_CTX
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        t0 = time.perf_counter()

        if self.provider == "llamacpp":
            model = _llama_model()
            with _LLAMA_INFER:
                out = model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.05,
                    max_tokens=max_tokens,
                )
            content = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
            content = _strip_thinking(content or "")
            return LLMResult(
                content=content,
                model="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
                latency_s=round(time.perf_counter() - t0, 2),
                provider="llamacpp",
            )

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_tokens},
            }
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "")
            if not content:
                content = (data.get("message") or {}).get("thinking", "")
            return LLMResult(
                content=content.strip(),
                model=data.get("model", self.model),
                latency_s=round(time.perf_counter() - t0, 2),
                provider="ollama",
            )

        if self.provider == "openai":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return LLMResult(
                content=data["choices"][0]["message"]["content"].strip(),
                model=data.get("model", self.model),
                latency_s=round(time.perf_counter() - t0, 2),
                provider="openai",
            )
        raise RuntimeError("LLM provider disabled")


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def llm_status() -> dict[str, Any]:
    client = get_client()
    status = client.health()
    status["base_url"] = (
        "本地 GGUF（llama.cpp）" if client.provider == "llamacpp" else client.base_url
    )
    return status


def warmup() -> dict[str, Any]:
    """Load the local model ahead of the first request (used by tests/demo)."""
    t0 = time.perf_counter()
    _llama_model()
    return {"loaded": True, "seconds": round(time.perf_counter() - t0, 2)}
