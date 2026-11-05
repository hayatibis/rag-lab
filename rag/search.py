"""4. adım: soruyu embed edip kosinüs benzerliğiyle en yakın k parçayı bulur."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag import config
from rag.embed import embed_texts, texts_digest
from rag.jsonl import read_jsonl


class IndexMissing(RuntimeError):
    pass


@dataclass
class Hit:
    id: str
    path: str
    heading_path: list[str]
    score: float
    text: str


@dataclass
class Index:
    vectors: np.ndarray
    chunks: list[dict]

    @classmethod
    def load(cls, data_dir: Path = config.DATA_DIR) -> "Index":
        npz_path, chunks_path = data_dir / "index.npz", data_dir / "chunks.jsonl"
        if not npz_path.exists() or not chunks_path.exists():
            raise IndexMissing(f"indeks yok ({data_dir}) — önce `uv run python index.py` çalıştır")
        data = np.load(npz_path)
        chunks = read_jsonl(chunks_path)
        # id'ler yalnız konumu anlatır; metin değişip embed atlanırsa özet yakalar
        same_ids = [str(i) for i in data["ids"]] == [c["id"] for c in chunks]
        same_text = ("digest" in data.files
                     and str(data["digest"]) == texts_digest([c["text"] for c in chunks]))
        if not (same_ids and same_text):
            raise IndexMissing("chunks.jsonl ile index.npz uyuşmuyor — "
                               "`uv run python index.py --step embed` çalıştır")
        return cls(data["vectors"], chunks)


def top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    # vektörler normalize ⇒ nokta çarpım = kosinüs benzerliği
    scores = vectors @ query
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def search(index: Index, question: str, k: int = config.TOP_K, embed_fn=embed_texts) -> list[Hit]:
    q = embed_fn([question])[0]
    hits = []
    for i, score in top_k(index.vectors, q, k):
        c = index.chunks[i]
        hits.append(Hit(c["id"], c["path"], c["heading_path"], score, c["text"]))
    return hits
