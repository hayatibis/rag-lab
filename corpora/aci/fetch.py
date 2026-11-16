"""Uçak kazası raporları korpusu: indir → OCR → soru seti.

manifest.json'daki raporlar (hayatibis.dev'deki Air Crash Investigation indeksinden
seçilmiş 25 resmî rapor) data/aci/ altına iner. Hiçbiri repoya girmez: data/ gitignore'da.

    uv run python corpora/aci/fetch.py              # hepsi: download → ocr → eval → stats
    uv run python corpora/aci/fetch.py --step ocr   # tek adım

Sonra iki indeks kurulup kıyaslanır (README'deki "Uçak kazası raporları" bölümü):
    RAG_SOURCE=data/aci/pdf RAG_DATA=data/aci/index-raw uv run python index.py
    RAG_SOURCE=data/aci/ocr RAG_DATA=data/aci/index-ocr uv run python index.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pypdf import PdfReader  # noqa: E402

from rag.load import pdf_to_text  # noqa: E402

OUT = PROJECT_ROOT / "data" / "aci"
PDF_DIR, OCR_DIR = OUT / "pdf", OUT / "ocr"
USER_AGENT = "rag-lab/0.1 (research; framework-free RAG demo)"

# Korpusta olmayan kazalar: doğru cevap "bulamadım".
TRAPS = [
    "What caused the Tenerife airport disaster in 1977?",
    "Why did Turkish Airlines Flight 1951 crash near Amsterdam?",
]


def load_manifest() -> list[dict]:
    return json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def download(reports: list[dict]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for r in reports:
        dest = PDF_DIR / f"{r['id']}.pdf"
        if dest.exists():
            continue
        req = urllib.request.Request(r["report_url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
        except Exception as e:  # bir rapor inmezse diğerleri devam etsin
            print(f"İNMEDİ {r['id']}: {type(e).__name__}: {e}")
            continue
        if not body.startswith(b"%PDF"):
            print(f"PDF DEĞİL {r['id']}: {body[:60]!r}")
            continue
        dest.write_bytes(body)
        print(f"indi {r['id']} ({len(body) / 1e6:.1f} MB)")


def ocr(reports: list[dict]) -> None:
    """Metni olmayan (taranmış) sayfalara OCR; metni olan sayfalara dokunmaz (--skip-text)."""
    if shutil.which("ocrmypdf") is None:
        raise SystemExit("ocrmypdf yok — `brew install ocrmypdf`")
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    for r in reports:
        src, dest = PDF_DIR / f"{r['id']}.pdf", OCR_DIR / f"{r['id']}.pdf"
        if not src.exists() or dest.exists():
            continue
        t0 = time.monotonic()
        proc = subprocess.run(
            ["ocrmypdf", "--skip-text", "--language", "eng", "--output-type", "pdf",
             "--optimize", "0", "--quiet", str(src), str(dest)],
            capture_output=True, text=True)
        took = time.monotonic() - t0
        if proc.returncode != 0:
            print(f"OCR HATA {r['id']} (çıkış {proc.returncode}): {proc.stderr.strip()[-300:]}")
            continue
        print(f"ocr {r['id']} {took:.0f} sn")


def write_eval(reports: list[dict]) -> None:
    """Rapor başına iki soru: adıyla (kolay) ve yalnız nedeniyle (adsız, zor)."""
    rows = []
    for r in reports:
        path = f"{r['id']}.pdf"
        if not (PDF_DIR / path).exists():  # inmeyen raporu sormak ölçümü haksız düşürür
            continue
        name = r["incident"].split(" (")[0]
        rows.append({"question": f"What caused the accident of {name}?",
                     "expected_path": path, "expected_phrase": "", "kind": "named"})
        rows.append({"question": f"Which accident is this: {r['cause']}",
                     "expected_path": path, "expected_phrase": "", "kind": "cause"})
    for q in TRAPS:
        rows.append({"question": q, "expected_path": None,
                     "expected_phrase": "bulamadım", "kind": "trap"})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"eval: {len(rows)} soru → {OUT / 'eval.jsonl'}")


def stats(reports: list[dict]) -> None:
    """Rapor başına sayfa sayısı, OCR'sız ve OCR'lı çıkan metin miktarı."""
    lines = ["id\tsayfa\tmetinsiz_sayfa_ham\tkarakter_ham\tkarakter_ocr"]
    for r in reports:
        raw, done = PDF_DIR / f"{r['id']}.pdf", OCR_DIR / f"{r['id']}.pdf"
        if not raw.exists():
            lines.append(f"{r['id']}\t-\t-\t-\t-")
            continue
        pages = PdfReader(str(raw)).pages
        empty = sum(1 for p in pages if not (p.extract_text() or "").strip())
        chars_ocr = len(pdf_to_text(done)) if done.exists() else "-"
        lines.append(f"{r['id']}\t{len(pages)}\t{empty}\t{len(pdf_to_text(raw))}\t{chars_ocr}")
    report = "\n".join(lines) + "\n"
    (OUT / "stats.tsv").write_text(report, encoding="utf-8")
    print(report)


STEPS = {"download": download, "ocr": ocr, "eval": write_eval, "stats": stats}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Uçak kazası raporları korpusu")
    p.add_argument("--step", choices=list(STEPS))
    args = p.parse_args(argv)
    reports = load_manifest()
    for name, fn in STEPS.items():
        if args.step in (None, name):
            fn(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
