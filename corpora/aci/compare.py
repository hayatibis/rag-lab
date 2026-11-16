"""Aynı indeks ve aynı sorularla üç arama: dense (kosinüs), BM25 ve hibrit (RRF).

    RAG_DATA=data/aci/index-ocr RAG_EVAL=data/aci/eval.jsonl \
        uv run python corpora/aci/compare.py

Dosya seviyesinde hit@1 / hit@5 ve soru türüne göre (named / cause) döker; hit@1'de
ıskalayan her soru için yerine hangi raporun geldiğini de yazar (hangi kazalar karışıyor?).
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from rag import config  # noqa: E402
from rag.embed import embed_texts  # noqa: E402
from rag.jsonl import read_jsonl  # noqa: E402
from rag.search import Index  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
# Robertson ve Zaragoza (2009): "A common combination would be b = 0.5 and k1 = 2. (However, many
# experiments suggest a somewhat lower value of k1 and a somewhat higher value of b.)"
K1, B = 1.5, 0.75
RRF_K = 60  # Cormack, Clarke ve Büttcher (2009): pilot denemede sabitlenmiş, sonra değiştirilmemiş


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            for term, tf in d.items():
                self.postings[term].append((i, tf))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = K1 * (1 - B + B * self.lens[i] / self.avg)
                s[i] += idf * tf * (K1 + 1) / (tf + norm)
        return s


def ranked(scores: np.ndarray, n: int = 200) -> list[int]:
    return [int(i) for i in np.argsort(-scores)[:n]]


def rrf(*rankings: list[int]) -> list[int]:
    fused: dict[int, float] = defaultdict(float)
    for r in rankings:
        for pos, i in enumerate(r, 1):
            fused[i] += 1 / (RRF_K + pos)
    return sorted(fused, key=fused.get, reverse=True)


def file_rank(order: list[int], chunks: list[dict], expected: str) -> int | None:
    """Beklenen dosyanın ilk parçasının, dosya tekilleştirilmiş sıralamadaki yeri."""
    seen: list[str] = []
    for i in order:
        p = chunks[i]["path"]
        if p not in seen:
            seen.append(p)
        if p == expected:
            return len(seen)
    return None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold()


def chunk_rank(order: list[int], chunks: list[dict], expected: str, phrase: str) -> int | None:
    """Doğru dosyadan ve beklenen ifadeyi içeren ilk parçanın sırası (parça seviyesi)."""
    p = norm(phrase)
    for pos, i in enumerate(order, 1):
        if chunks[i]["path"] == expected and p in norm(chunks[i]["text"]):
            return pos
    return None


def top_file(order: list[int], chunks: list[dict]) -> str:
    return chunks[order[0]]["path"]


def main() -> int:
    index = Index.load()
    chunks = index.chunks
    cases = [c for c in read_jsonl(config.EVAL_PATH) if c.get("expected_path")]
    bm25 = BM25([c["text"] for c in chunks])
    qvecs = embed_texts([c["question"] for c in cases])

    methods = ("dense", "bm25", "hibrit")
    tally: dict[tuple[str, str, int], int] = Counter()
    totals: Counter = Counter()
    misses: list[str] = []
    for case, q in zip(cases, qvecs):
        d = ranked(index.vectors @ q)
        b = ranked(bm25.scores(case["question"]))
        orders = {"dense": d, "bm25": b, "hibrit": rrf(d, b)}
        kind = case.get("kind", "?")
        totals[kind] += 1
        for m, order in orders.items():
            rank = file_rank(order, chunks, case["expected_path"])
            for k in (1, 5):
                if rank is not None and rank <= k:
                    tally[(m, kind, k)] += 1
            if case.get("expected_phrase"):
                crank = chunk_rank(order, chunks, case["expected_path"], case["expected_phrase"])
                for k in (5, 8):
                    if crank is not None and crank <= k:
                        tally[(m, "parça", k)] += 1
                if crank is None or crank > 8:
                    misses.append(f"{m:6} parça {case['question'][:70]:70} ilk doğru parça "
                                  f"{crank or '-'}")
            if rank != 1:
                misses.append(f"{m:6} {kind:5} beklenen {case['expected_path']:40} "
                              f"gelen {top_file(order, chunks):40} sıra {rank or '-'}")

    print(f"{len(chunks)} parça, {len({c['path'] for c in chunks})} rapor, {len(cases)} soru\n")
    print(f"{'':8}" + "".join(f"{kind + ' @' + str(k):>12}" for kind in totals for k in (1, 5)))
    for m in methods:
        row = "".join(f"{tally[(m, kind, k)]:>9}/{totals[kind]:<2}" for kind in totals for k in (1, 5))
        print(f"{m:8}{row}")
    n_phrase = sum(1 for c in cases if c.get("expected_phrase"))
    if n_phrase:
        print(f"\nparça seviyesi (doğru rapor + beklenen ifade), {n_phrase} soru:")
        for m in methods:
            print(f"{m:8}" + "".join(f"   @{k}: {tally[(m, 'parça', k)]}/{n_phrase}" for k in (5, 8)))
    print("\nhit@1 ıskaları:")
    print("\n".join(misses) or "(yok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
