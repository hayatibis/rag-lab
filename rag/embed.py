"""3. adım: parçaları Ollama bge-m3 ile vektöre çevirir → data/index.npz."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx
import numpy as np

from rag import config
from rag.jsonl import read_jsonl


class OllamaError(RuntimeError):
    pass


def texts_digest(texts: list[str]) -> str:
    """Parça metinlerinin özeti; vektörler hangi metinden üretildi, sonradan doğrulanır."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()


def normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def embed_texts(texts: list[str], batch_size: int = 16,
                client: httpx.Client | None = None, progress: bool = False) -> np.ndarray:
    client = client or httpx.Client(base_url=config.OLLAMA_URL, timeout=300)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            r = client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": batch})
        except httpx.ConnectError as e:
            raise OllamaError(
                f"Ollama çalışmıyor ({config.OLLAMA_URL}) — `ollama serve` çalıştır") from e
        if r.status_code == 404:
            raise OllamaError(f"model yok — `ollama pull {config.EMBED_MODEL}` çalıştır")
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        if progress:
            print(f"\rembed: {len(out)}/{len(texts)}", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return normalize_rows(np.array(out, dtype=np.float32).reshape(len(out), -1))


def run(data_dir: Path = config.DATA_DIR) -> np.ndarray:
    chunks = read_jsonl(data_dir / "chunks.jsonl")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, progress=True)
    np.savez(data_dir / "index.npz", vectors=vectors, ids=np.array([c["id"] for c in chunks]),
             digest=np.array(texts_digest(texts)))
    print(f"embed: {vectors.shape[0]} vektör × {vectors.shape[1]} boyut → {data_dir / 'index.npz'}")
    return vectors
