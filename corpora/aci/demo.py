"""Uçak kazası korpusu için yayınlanabilir, önceden hesaplanmış demo üretir.

    RAG_DATA=data/aci/index-ocr uv run --env-file .env python corpora/aci/demo.py
    RAG_DATA=data/aci/index-ocr uv run python corpora/aci/demo.py --retrieve-only

Çıktı data/aci/demo/ altına: traces/qNN.html (rag-lab'in iz sayfası, her soru için)
ve demo.json (galeri sayfasının verisi: korpus, OCR etkisi, üç arama yöntemi, skor eşiği).
Canlı sunucu yok: sorular, skorlar ve cevaplar burada bir kez hesaplanır.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from compare import BM25, chunk_rank, file_rank, ranked, rrf  # noqa: E402
from rag import config, trace  # noqa: E402
from rag.embed import embed_texts  # noqa: E402
from rag.jsonl import read_jsonl  # noqa: E402
from rag.search import Index  # noqa: E402

OUT = PROJECT_ROOT / "data" / "aci" / "demo"
EVAL_HAND = HERE / "eval-hand.jsonl"
STATS = PROJECT_ROOT / "data" / "aci" / "stats.tsv"

# Galeride gösterilecek sorular: her biri bir dersi taşır.
SELECTED = [
    ("Which engine did the British Midland 737 crew shut down at Kegworth?", "Taranmış 1990 raporu, OCR'dan sonra bulunuyor"),
    ("What did the Kegworth pilots not hear about that passengers and cabin crew had seen?", "Cevap tek bir kısa satırda"),
    ("How many angle-of-attack sensors did MCAS rely on?", "Kısa, net bir olgu"),
    ("What style of management for captains did the NTSB recommend after United 173?", "Nadir ifade: BM25 buluyor, dense kaçırıyor"),
    ("Why did the AF447 crew never apply a stall recovery?", "Doğru rapor, yanlış parça"),
    ("Which cue rose on the airspeed display while the Colgan crew wasn't watching?", "Doğru rapor, cevap parçası çok geride"),
    ("At which lap joint did the Aloha 737 fuselage fail?", "Kod gibi bir terim: S-10L"),
    ("Which maintenance contractor failed to properly prepare the oxygen generators on ValuJet 592?", "Cevap sayfasının fontu bozuk, ama ad raporun başka sayfalarında geçiyor"),
    ("Germanwings'te kaptan kokpite neden giremedi?", "Türkçe soru, İngilizce rapor"),
    ("747'nin kargo kapısı uçuş sırasında neden açıldı?", "Türkçe soru, İngilizce rapor"),
    ("Why did Ethiopian Airlines Flight 302 crash after takeoff from Addis Ababa?", "Tuzak: korpusta yok, Lion Air'e benziyor"),
    ("Why did United Airlines Flight 232 lose all hydraulics over Iowa?", "Tuzak: korpusta yok, rapor linki inmedi"),
]


def decade_folder(path: str, manifest: dict[str, dict]) -> str:
    year = int(manifest[path.removesuffix(".pdf")]["date"][:4])
    return f"{year // 10 * 10}'ler/{path}"


def with_decades(index: Index, manifest: dict[str, dict]) -> Index:
    """İz sayfası renkleri klasörden alır; korpusu on yıllara bölünmüş klasörler gibi gösterir."""
    chunks = [{**c, "path": decade_folder(c["path"], manifest)} for c in index.chunks]
    return replace(index, chunks=chunks)


def top_files(order: list[int], chunks: list[dict], n: int = 5) -> list[str]:
    seen: list[str] = []
    for i in order:
        p = chunks[i]["path"]
        if p not in seen:
            seen.append(p)
        if len(seen) == n:
            break
    return seen


def read_stats() -> list[dict]:
    rows = []
    lines = STATS.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        rid, pages, empty, raw, ocr = line.split("\t")
        if pages == "-":
            continue
        rows.append({"id": rid, "pages": int(pages), "empty_raw": int(empty),
                     "chars_raw": int(raw), "chars_ocr": int(ocr)})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Uçak kazası demo sayfaları")
    p.add_argument("--retrieve-only", action="store_true", help="Claude'u çağırma")
    p.add_argument("--site", action="store_true",
                   help="yayınlanacak kopyayı repodaki site/ klasörüne de yaz (Vercel buradan yayınlar)")
    args = p.parse_args(argv)

    manifest = {r["id"]: r for r in json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))}
    index = Index.load()
    chunks = index.chunks
    traced_index = with_decades(index, manifest)
    cases = list(read_jsonl(EVAL_HAND))
    by_q = {c["question"]: c for c in cases}
    bm25 = BM25([c["text"] for c in chunks])

    # Tüm sorular: en yüksek kosinüs skoru (gerçek soru mu, tuzak mı?) ve üç yöntemin sırası
    qvecs = embed_texts([c["question"] for c in cases])
    all_rows = []
    for case, q in zip(cases, qvecs):
        dense_scores = index.vectors @ q
        d, b = ranked(dense_scores), ranked(bm25.scores(case["question"]))
        orders = {"dense": d, "bm25": b, "hybrid": rrf(d, b)}
        row = {"question": case["question"], "kind": case["kind"], "lang": case["lang"],
               "expected": case["expected_path"], "max_score": round(float(dense_scores.max()), 4),
               "top_file": chunks[d[0]]["path"]}
        if case["expected_path"]:
            for m, order in orders.items():
                row[f"{m}_file"] = file_rank(order, chunks, case["expected_path"])
                row[f"{m}_chunk"] = chunk_rank(order, chunks, case["expected_path"],
                                               case["expected_phrase"])
        all_rows.append(row)

    # Seçilen sorular: iz sayfası + üç yöntemin ilk 5 raporu
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    gallery = []
    for n, (question, lesson) in enumerate(SELECTED, 1):
        case = by_q[question]
        t = trace.run_traced(traced_index, question, retrieve_only=args.retrieve_only)
        name = f"q{n:02d}.html"
        (OUT / "traces" / name).write_text(trace.render_html(t), encoding="utf-8")
        q = embed_texts([question])[0]
        d, b = ranked(index.vectors @ q), ranked(bm25.scores(question))
        row = next(r for r in all_rows if r["question"] == question)
        gallery.append({
            "n": n, "question": question, "lesson": lesson, "kind": case["kind"],
            "expected": case["expected_path"], "expected_page": case["expected_page"],
            "expected_phrase": case["expected_phrase"], "trace": f"traces/{name}",
            "max_score": row["max_score"],
            "answer": t["generate"]["answer"] if t["generate"] else None,
            "methods": {m: {"files": top_files(o, chunks),
                            "file_rank": row.get(f"{m}_file"), "chunk_rank": row.get(f"{m}_chunk")}
                        for m, o in {"dense": d, "bm25": b, "hybrid": rrf(d, b)}.items()},
        })
        print(f"{name}  {question}")

    def summary(rows: list[dict], key: str, k: int) -> int:
        return sum(1 for r in rows if r.get(key) is not None and r[key] <= k)

    scored = [r for r in all_rows if r["expected"]]
    data = {
        "corpus": {"reports": len({c["path"] for c in chunks}), "chunks": len(chunks),
                   "dim": int(index.vectors.shape[1]), "embed_model": config.EMBED_MODEL,
                   "chunk_tokens": config.CHUNK_TOKENS, "overlap_tokens": config.OVERLAP_TOKENS,
                   "by_decade": dict(Counter(decade_folder(c["path"], manifest).split("/")[0]
                                             for c in chunks))},
        "reports": [{"id": r["id"], "incident": r["incident"], "date": r["date"], "agency": r["agency"],
                     "report_id": r["report_id"], "report_url": r["report_url"]}
                    for r in manifest.values()],
        "stats": read_stats(),
        "results": {m: {"file@1": summary(scored, f"{m}_file", 1),
                        "chunk@5": summary(scored, f"{m}_chunk", 5),
                        "chunk@8": summary(scored, f"{m}_chunk", 8)}
                    for m in ("dense", "bm25", "hybrid")} | {"n": len(scored)},
        "scores": [{"question": r["question"], "kind": r["kind"], "max_score": r["max_score"],
                    "top_file": r["top_file"]} for r in all_rows],
        "gallery": gallery,
        "generated_with_claude": not args.retrieve_only,
    }
    (OUT / "demo.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # '<' kaçışlanır: veri içindeki "</script>" sayfayı bozamaz
    blob = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = (HERE / "demo_template.html").read_text(encoding="utf-8").replace("/*__DEMO__*/null", blob)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"→ {OUT / 'index.html'}")

    if args.site:
        site = PROJECT_ROOT / "site"
        (site / "traces").mkdir(parents=True, exist_ok=True)
        # şablon gövdeden başlar (Artifact kendi iskeletini ekler); tek başına yayında iskelet gerekir
        head = ('<!doctype html>\n<html lang="tr">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
        (site / "index.html").write_text(head + page, encoding="utf-8")
        for f in sorted((OUT / "traces").glob("q*.html")):
            (site / "traces" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"→ {site}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
