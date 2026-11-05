"""Bir sorunun RAG boru hattındaki yolculuğunu kaydeder ve tek dosyalık HTML'e çizer."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rag import config
from rag.answer import SYSTEM_PROMPT, build_prompt, generate
from rag.embed import embed_texts
from rag.search import Hit, Index, top_k

TEMPLATE = Path(__file__).with_name("trace_template.html")
RUNNERS_UP = 4  # kesimin hemen altında kalan, "kıl payı" parçalar


def pca_2d(vectors: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """1024 boyutu en çok yayılan 2 eksene indirir; oran = bu 2 eksenin taşıdığı varyans payı."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2]
    var = s ** 2
    ratio = float(var[:2].sum() / var.sum()) if var.sum() > 0 else 0.0
    return centered @ comps.T, (extra - mean) @ comps.T, ratio


def score_histogram(scores: np.ndarray, bins: int = 40) -> dict:
    counts, edges = np.histogram(scores, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def folder_of(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "(kök)"


def _hit_dict(rank: int, i: int, score: float, index: Index, pts: np.ndarray) -> dict:
    c = index.chunks[i]
    return {"rank": rank, "i": i, "id": c["id"], "path": c["path"], "heading_path": c["heading_path"],
            "folder": folder_of(c["path"]), "score": score, "text": c["text"],
            "x": float(pts[i, 0]), "y": float(pts[i, 1])}


def run_traced(index: Index, question: str, k: int = config.TOP_K,
               embed_fn=embed_texts, generate_fn=generate, retrieve_only: bool = False) -> dict:
    steps: list[dict] = []

    t0 = time.perf_counter()
    q = embed_fn([question])[0]
    steps.append({"name": "embed", "ms": (time.perf_counter() - t0) * 1000})

    t0 = time.perf_counter()
    scores = index.vectors @ q
    ranked = top_k(index.vectors, q, k + RUNNERS_UP)
    steps.append({"name": "search", "ms": (time.perf_counter() - t0) * 1000})

    pts, q_xy, ratio = pca_2d(index.vectors, q[None, :])
    top = ranked[:k]
    runners = ranked[k:]
    hits = [_hit_dict(r, i, s, index, pts) for r, (i, s) in enumerate(top, 1)]
    hit_objs = [Hit(h["id"], h["path"], h["heading_path"], h["score"], h["text"]) for h in hits]

    t0 = time.perf_counter()
    user_prompt = build_prompt(question, hit_objs)
    steps.append({"name": "prompt", "ms": (time.perf_counter() - t0) * 1000})

    gen = None
    if not retrieve_only:
        t0 = time.perf_counter()
        text, usage = generate_fn(question, hit_objs)
        steps.append({"name": "generate", "ms": (time.perf_counter() - t0) * 1000})
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
        gen = {"answer": text, "usage": usage, "cited": cited, "model": config.CLAUDE_MODEL}

    # renk klasörü takip eder: sıra korpustaki parça sayısına göre sabit, sorudan bağımsız
    folder_counts = Counter(folder_of(c["path"]) for c in index.chunks)
    folders = [f for f, _ in folder_counts.most_common()]
    fidx = {f: n for n, f in enumerate(folders)}
    hit_rank = {h["i"]: h["rank"] for h in hits}
    points = [[round(float(pts[i, 0]), 4), round(float(pts[i, 1]), 4), round(float(scores[i]), 4),
               fidx[folder_of(c["path"])], hit_rank.get(i, 0)]
              for i, c in enumerate(index.chunks)]
    labels = [" › ".join([c["path"], *c["heading_path"]])[:160] for c in index.chunks]

    return {
        "question": question,
        "created": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "steps": steps,
        "corpus": {"n_docs": len({c["path"] for c in index.chunks}), "n_chunks": len(index.chunks),
                   "dim": int(index.vectors.shape[1]), "embed_model": config.EMBED_MODEL,
                   "folders": [{"name": f, "chunks": folder_counts[f]} for f in folders]},
        "embed": {"dim": int(q.shape[0]), "preview": [round(float(x), 4) for x in q],
                  "norm": float(np.linalg.norm(q))},
        "search": {"hits": hits,
                   "runners_up": [_hit_dict(r, i, s, index, pts)
                                  for r, (i, s) in enumerate(runners, k + 1)],
                   "kth_score": hits[-1]["score"] if hits else None,
                   "next_score": runners[0][1] if runners else None,
                   "max_score": float(scores.max()), "mean_score": float(scores.mean()),
                   "histogram": score_histogram(scores)},
        "map": {"points": points, "labels": labels, "question": [float(q_xy[0, 0]), float(q_xy[0, 1])],
                "variance_ratio": ratio},
        "prompt": {"system": SYSTEM_PROMPT, "user": user_prompt,
                   "parts": [{"kind": "system", "chars": len(SYSTEM_PROMPT)}]
                   + [{"kind": "chunk", "rank": h["rank"], "folder": h["folder"],
                       "chars": len(h["text"]) + 30} for h in hits]
                   + [{"kind": "question", "chars": len(question) + 8}]},
        "generate": gen,
    }


def render_html(trace: dict) -> str:
    # '<' kaçışlanır: veri içindeki "</script>" sayfayı bozamaz
    data = json.dumps(trace, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__TRACE__*/null", data)


def save(trace: dict, data_dir: Path = config.DATA_DIR) -> Path:
    out_dir = data_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stem}.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    path = out_dir / f"{stem}.html"
    path.write_text(render_html(trace), encoding="utf-8")
    return path
