"""eval.jsonl'daki sorularla arama (hit@k) ve cevap isabetini ölçer."""
import argparse
import sys
from datetime import datetime

from rag import config
from rag.answer import TRANSIENT_ERRORS, MissingApiKey, answer
from rag.embed import OllamaError
from rag.jsonl import read_jsonl
from rag.search import Hit, Index, IndexMissing, search

EVAL_PATH = config.PROJECT_ROOT / "eval.jsonl"


def score_case(case: dict, hits: list[Hit], answer_text: str | None) -> dict:
    expected = case.get("expected_path")
    rank = next((n for n, h in enumerate(hits, 1) if h.path == expected), None)
    return {
        "question": case["question"],
        "hit": None if expected is None else rank is not None,
        "rank": rank,
        "answer_ok": None if answer_text is None
        else case["expected_phrase"].casefold() in answer_text.casefold(),
    }


def _ratio(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "—"


def summarize(rows: list[dict]) -> str:
    return f"hit@k: {_ratio(rows, 'hit')}   cevap: {_ratio(rows, 'answer_ok')}"


def _mark(v) -> str:
    return {True: "✓", False: "✗", None: "·"}[v]


def run_case(case: dict, k: int, retrieve_only: bool,
             search_fn, answer_fn) -> tuple[dict, list[str]]:
    """Tek soruyu ölçer; geçici bir API/ağ hatası yalnız bu soruyu başarısız sayar."""
    error = None
    hits: list[Hit] = []
    text = None
    try:
        hits = search_fn(case["question"], k)
        if not retrieve_only:
            text = answer_fn(case["question"], hits)
    except TRANSIENT_ERRORS as e:
        error = f"{type(e).__name__}: {e}"
        if not retrieve_only:
            text = ""  # cevap alınamadı ⇒ isabetsiz sayılır
    r = score_case(case, hits, text)
    lines = [f"{_mark(r['hit'])} arama (sıra {r['rank'] or '-'})  "
             f"{_mark(r['answer_ok'])} cevap  {r['question']}"]
    if error:
        lines.append(f"    HATA {error}")
    elif text is not None:
        lines.append("    → " + text.replace("\n", " ")[:200])
    return r, lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RAG değerlendirmesi")
    p.add_argument("-k", type=int, default=config.TOP_K)
    p.add_argument("--retrieve-only", action="store_true")
    args = p.parse_args(argv)
    try:
        index = Index.load()
        rows, lines = [], []
        for case in read_jsonl(EVAL_PATH):
            r, case_lines = run_case(case, args.k, args.retrieve_only,
                                     search_fn=lambda q, k: search(index, q, k),
                                     answer_fn=answer)
            rows.append(r)
            lines.extend(case_lines)
    except (IndexMissing, OllamaError, MissingApiKey) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    report = "\n".join(lines) + f"\n\n{summarize(rows)}  (k={args.k})\n"
    out = config.DATA_DIR / f"eval_{datetime.now():%Y%m%d-%H%M}.txt"
    out.write_text(report, encoding="utf-8")
    print(report + f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
