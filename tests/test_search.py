import numpy as np
import pytest

from rag.embed import normalize_rows, texts_digest
from rag.jsonl import write_jsonl
from rag.search import Index, IndexMissing, search, top_k


def test_top_k_orders_by_score_and_caps_k():
    v = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
    res = top_k(v, np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert [i for i, _ in res] == [0, 2, 1]
    assert res[0][1] == pytest.approx(1.0)


def test_index_load_missing_gives_instruction(tmp_path):
    with pytest.raises(IndexMissing, match="index.py"):
        Index.load(tmp_path)


def test_index_load_detects_mismatch(tmp_path):
    np.savez(tmp_path / "index.npz", vectors=np.zeros((1, 2), dtype=np.float32),
             ids=np.array(["a#0"]))
    write_jsonl(tmp_path / "chunks.jsonl",
                [{"id": "b#0", "path": "b", "heading_path": [], "text": "t"}])
    with pytest.raises(IndexMissing, match="uyuşmuyor"):
        Index.load(tmp_path)


def _write_index(tmp_path, texts_at_embed_time, texts_now):
    ids = [f"{i}#0" for i in range(len(texts_now))]
    np.savez(tmp_path / "index.npz", vectors=np.eye(len(ids), dtype=np.float32),
             ids=np.array(ids), digest=np.array(texts_digest(texts_at_embed_time)))
    write_jsonl(tmp_path / "chunks.jsonl", [
        {"id": i, "path": i, "heading_path": [], "text": t} for i, t in zip(ids, texts_now)
    ])


def test_index_load_roundtrip(tmp_path):
    _write_index(tmp_path, ["ta", "tb"], ["ta", "tb"])
    assert Index.load(tmp_path).vectors.shape == (2, 2)


def test_index_load_detects_edited_text_with_same_ids(tmp_path):
    _write_index(tmp_path, ["ta", "tb"], ["ta", "tb DÜZENLENDİ"])
    with pytest.raises(IndexMissing, match="uyuşmuyor"):
        Index.load(tmp_path)


def test_search_maps_hits():
    index = Index(
        vectors=normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]])),
        chunks=[
            {"id": "a#0", "path": "a", "heading_path": ["H"], "text": "ta"},
            {"id": "b#0", "path": "b", "heading_path": [], "text": "tb"},
        ],
    )
    hits = search(index, "soru", k=1, embed_fn=lambda texts: np.array([[0.0, 1.0]], dtype=np.float32))
    assert len(hits) == 1
    assert hits[0].id == "b#0" and hits[0].text == "tb"
    assert hits[0].score == pytest.approx(1.0)
