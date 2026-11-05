import numpy as np
import pytest

from rag.embed import normalize_rows
from rag.search import Index
from rag.trace import folder_of, pca_2d, render_html, run_traced, score_histogram


def fake_index() -> Index:
    vecs = normalize_rows(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1], [0, 0, 1]],
                                   dtype=float))
    chunks = [{"id": f"{p}#0", "path": p, "heading_path": ["H"], "text": f"metin {i}"}
              for i, p in enumerate(["notlar/a.md", "notlar/b.md", "raporlar/c.md", "raporlar/d.md", "e.md"])]
    return Index(vecs, chunks)


def fake_embed(texts):
    return normalize_rows(np.array([[1.0, 0.05, 0.0]]))


def fake_generate(question, hits):
    return "Cevap burada [1] ve şurada [2][1].", {"input_tokens": 900, "output_tokens": 40}


def test_pca_2d_keeps_main_direction_and_projects_extra_consistently():
    v = np.array([[-2.0, 0, 0], [-1, 0.1, 0], [1, -0.1, 0], [2, 0, 0]])
    pts, extra, ratio = pca_2d(v, np.array([[2.0, 0, 0]]))
    assert pts.shape == (4, 2) and extra.shape == (1, 2)
    assert 0.99 < ratio <= 1.0  # neredeyse tüm varyans ilk eksende
    assert np.allclose(extra[0], pts[3], atol=1e-6)
    assert abs(pts[0, 0] - pts[3, 0]) > 10 * abs(pts[0, 1] - pts[3, 1])


def test_score_histogram_counts_everything():
    h = score_histogram(np.array([0.1, 0.2, 0.2, 0.9]), bins=4)
    assert sum(h["counts"]) == 4 and len(h["edges"]) == 5


def test_folder_of():
    assert folder_of("raporlar/raporlar/x.html") == "raporlar"
    assert folder_of("e.md") == "(kök)"


def test_run_traced_records_every_step():
    t = run_traced(fake_index(), "Soru?", k=2, embed_fn=fake_embed, generate_fn=fake_generate)
    assert [s["name"] for s in t["steps"]] == ["embed", "search", "prompt", "generate"]
    assert all(s["ms"] >= 0 for s in t["steps"])
    assert t["embed"]["dim"] == 3 and len(t["embed"]["preview"]) == 3
    hits = t["search"]["hits"]
    assert [h["path"] for h in hits] == ["notlar/a.md", "notlar/b.md"]
    assert hits[0]["rank"] == 1 and hits[0]["score"] >= hits[1]["score"]
    assert t["search"]["kth_score"] == pytest.approx(hits[1]["score"])
    assert t["search"]["next_score"] < t["search"]["kth_score"]
    assert len(t["map"]["points"]) == 5 and len(t["map"]["question"]) == 2
    assert sum(t["search"]["histogram"]["counts"]) == 5
    assert t["prompt"]["user"].endswith("Soru: Soru?")
    assert t["generate"]["cited"] == [1, 2]
    assert t["generate"]["usage"]["input_tokens"] == 900


def test_run_traced_retrieve_only_has_no_generate_step():
    t = run_traced(fake_index(), "Soru?", k=2, embed_fn=fake_embed, generate_fn=fake_generate,
                   retrieve_only=True)
    assert [s["name"] for s in t["steps"]] == ["embed", "search", "prompt"]
    assert t["generate"] is None


def test_render_html_embeds_data_safely():
    idx = fake_index()
    idx.chunks[0]["text"] = "kötü </script><script>alert(1)"
    t = run_traced(idx, "Soru?", k=2, embed_fn=fake_embed, generate_fn=fake_generate)
    page = render_html(t)
    assert "Soru?" in page
    assert page.count("</script>") == page.count("<script")
    assert "alert(1)" in page  # veri kaybolmadı, yalnız kaçışlandı
