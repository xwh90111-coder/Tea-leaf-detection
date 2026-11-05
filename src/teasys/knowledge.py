"""D2 - tea knowledge base (document chunking, vector index, hybrid retrieval).

The index is a persistent local vector store:

    data/knowledge_index/chunks.json      chunk text + metadata
    data/knowledge_index/embeddings.npy   L2-normalised dense vectors
    data/knowledge_index/index_meta.json  build manifest

Dense retrieval uses BAAI/bge-small-zh-v1.5 (Chinese, 512-d) through
transformers; if the model is unavailable a TF-IDF character n-gram fallback
keeps the system functional.  Dense and BM25 rankings are fused with
Reciprocal Rank Fusion (RRF), so both semantic and keyword matches contribute.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config

_INDEX_DIR = config.DATA_DIR / "knowledge_index"
_CHUNKS_FILE = _INDEX_DIR / "chunks.json"
_EMB_FILE = _INDEX_DIR / "embeddings.npy"
_META_FILE = _INDEX_DIR / "index_meta.json"

_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_section(body: str, size: int, overlap: int) -> list[str]:
    sents = _sentences(body)
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= size:
            cur += s
            continue
        if cur:
            chunks.append(cur.strip())
        # start new chunk with tail overlap from previous text
        tail = cur[-overlap:] if overlap and cur else ""
        cur = (tail + s) if len(s) < size else s[:size]
        while len(cur) > size:
            chunks.append(cur[:size].strip())
            cur = cur[size - overlap :]
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) >= 40]


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_title = path.stem
        sections: list[tuple[str, str]] = []
        current_heading = "概述"
        buf: list[str] = []
        for line in raw.splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                continue
            if line.startswith("## "):
                if buf:
                    sections.append((current_heading, "\n".join(buf)))
                current_heading = line[3:].strip()
                buf = []
                continue
            buf.append(line)
        if buf:
            sections.append((current_heading, "\n".join(buf)))

        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            for i, piece in enumerate(
                _chunk_section(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                text = f"《{doc_title}》{heading}：{piece}"
                cid = hashlib.md5(f"{path.name}:{heading}:{i}".encode()).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        id=cid,
                        doc=path.name,
                        title=doc_title,
                        section=heading,
                        text=text,
                        source=f"{path.name}#{heading}",
                    )
                )
    return chunks


def _docs_fingerprint() -> str:
    h = hashlib.md5()
    for p in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_mtime_ns).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class _BGEEmbedder:
    name = config.EMBEDDING_MODEL

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        local = config.ROOT / "models" / "bge-small-zh-v1.5"
        model_path = str(local if (local / "config.json").exists() else config.EMBEDDING_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.device = "cpu"

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**inputs).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                out.append(hidden.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 512), dtype=np.float32)


class _TfidfEmbedder:
    name = "tfidf-char-ngram (fallback)"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.meta: dict[str, Any] = {}
        self._embedder = None
        self._bm25 = None
        self._tokenized: list[list[str]] = []
        self._load_or_build()

    # -- build ------------------------------------------------------------
    def _load_or_build(self, force: bool = False) -> None:
        fingerprint = _docs_fingerprint()
        fresh = (
            not force
            and _CHUNKS_FILE.exists()
            and _EMB_FILE.exists()
            and _META_FILE.exists()
            and json.loads(_META_FILE.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
        )
        if fresh:
            data = json.loads(_CHUNKS_FILE.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**c) for c in data]
            self.embeddings = np.load(_EMB_FILE)
            self.meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
            if self.meta.get("backend", "").startswith("tfidf"):
                self._init_tfidf(fit=False)
            else:
                self._init_bge()
            self._init_bm25()
            return
        self.build(force=force)

    def _init_bge(self) -> None:
        try:
            self._embedder = _BGEEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[knowledge] BGE unavailable ({e}); falling back to TF-IDF")
            self._embedder = None

    def _init_tfidf(self, fit: bool) -> None:
        self._embedder = _TfidfEmbedder()
        if fit:
            self._embedder.fit([c.text for c in self.chunks])

    def _init_bm25(self) -> None:
        import jieba
        from rank_bm25 import BM25Okapi

        self._tokenized = [list(jieba.cut(c.text)) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def build(self, force: bool = False) -> dict[str, Any]:
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.chunks = load_chunks()
        if not self.chunks:
            self.embeddings = np.zeros((0, 512), dtype=np.float32)
            return {"chunks": 0}
        self._init_bge()
        texts = [c.text for c in self.chunks]
        if self._embedder is not None:
            emb = self._embedder.encode(texts)
            backend = f"bge:{config.EMBEDDING_MODEL} (transformers, CLS pooling)"
        else:
            self._init_tfidf(fit=True)
            emb = self._embedder.encode(texts)  # type: ignore[union-attr]
            backend = "tfidf-char-ngram (fallback)"
        self.embeddings = emb
        np.save(_EMB_FILE, emb)
        _CHUNKS_FILE.write_text(
            json.dumps([c.as_dict() for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self.meta = {
            "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "chunks": len(self.chunks),
            "docs": len({c.doc for c in self.chunks}),
            "dim": int(emb.shape[1]) if emb.size else 0,
            "backend": backend,
            "fingerprint": _docs_fingerprint(),
            "query_instruction": _QUERY_INSTRUCTION,
        }
        _META_FILE.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._init_bm25()
        return self.meta

    # -- search -----------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        if isinstance(self._embedder, _TfidfEmbedder):
            q = self._embedder.encode([query])[0]
        else:
            q = self._embedder.encode([_QUERY_INSTRUCTION + query])[0]  # type: ignore[union-attr]
        return q

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        k = k or config.TOP_K
        if not self.chunks or self.embeddings is None or self.embeddings.size == 0:
            return []
        with self._lock:
            q = self._embed_query(query)
            dense_scores = self.embeddings @ q
            dense_order = np.argsort(-dense_scores)[: max(20, k * 4)]

            bm25_scores = None
            if self._bm25 is not None:
                import jieba

                bm25_scores = np.array(self._bm25.get_scores(list(jieba.cut(query))))
                bm25_order = np.argsort(-bm25_scores)[: max(20, k * 4)]
            else:
                bm25_order = np.array([], dtype=int)

            # Reciprocal Rank Fusion
            rrf: dict[int, float] = {}
            for rank, idx in enumerate(dense_order):
                rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (60 + rank + 1)
            for rank, idx in enumerate(bm25_order):
                rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (60 + rank + 1)
            ranked = sorted(rrf.items(), key=lambda x: -x[1])[:k]

        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            results.append(
                {
                    "id": c.id,
                    "title": c.title,
                    "section": c.section,
                    "text": c.text,
                    "source": c.source,
                    "score": round(float(score), 6),
                    "dense_score": round(float(dense_scores[idx]), 4),
                    "bm25_score": round(float(bm25_scores[idx]), 3) if bm25_scores is not None else None,
                }
            )
        return results

    def status(self) -> dict[str, Any]:
        return {
            **self.meta,
            "docs_files": len(list(config.KNOWLEDGE_DIR.glob("*.md"))),
            "backend": self.meta.get("backend", "not built"),
        }

    def rebuild(self) -> dict[str, Any]:
        return self.build(force=True)


_KB: KnowledgeBase | None = None
_KB_LOCK = threading.Lock()


def get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        with _KB_LOCK:
            if _KB is None:
                _KB = KnowledgeBase()
    return _KB


def kb_meta() -> dict[str, Any]:
    """Read the index manifest without loading the embedding model."""
    if _META_FILE.exists():
        try:
            return json.loads(_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"chunks": 0, "docs": 0, "backend": "未构建", "built_at": "—"}
