"""1. adım: kaynak dosyaları düz metne çevirir → data/docs.jsonl."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from rag import config
from rag.jsonl import write_jsonl

BLOCK_TAGS = ["p", "div", "li", "section", "article", "pre", "blockquote",
              "h4", "h5", "h6", "table", "ul", "ol"]


@dataclass
class Doc:
    path: str
    type: str
    text: str
    n_chars: int


def should_include(rel: Path) -> bool:
    if rel.suffix.lower() not in config.INCLUDE_SUFFIXES:
        return False
    if rel.name.endswith(config.EXCLUDE_NAME_SUFFIXES):
        return False
    posix = "/" + rel.as_posix()
    return not any(f"/{d}/" in posix for d in config.EXCLUDE_DIRS)


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and should_include(path.relative_to(root)):
            yield path


def clean_text(text: str) -> str:
    """Satır içi boşlukları sadeleştirir, ardışık boş satırları teke indirir."""
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t ]+", " ", raw).strip()
        if line:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "svg", "noscript", "head", "img"]):
        tag.decompose()
    # HTML'de kaynak satır sonu = boşluk (pre hariç); yoksa satır başındaki '#' başlık sanılır
    for s in soup.find_all(string=True):
        if "\n" in s and s.find_parent("pre") is None:
            s.replace_with(re.sub(r"\s*\n\s*", " ", str(s)))
    # kod blokları çitlenir: içlerindeki '# yorum' satırları başlık sanılmaz
    for pre in soup.find_all("pre"):
        pre.replace_with(f"\n\n```\n{pre.get_text().strip()}\n```\n\n")
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        # "| a | b |": satır '#' hücresiyle başlasa bile başlık sanılmaz
        tr.replace_with("\n| " + " | ".join(cells) + " |\n")
    for level in (1, 2, 3):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n\n{'#' * level} {h.get_text(' ', strip=True)}\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
    for block in soup.find_all(BLOCK_TAGS):
        block.insert_before("\n\n")
    return clean_text(soup.get_text())


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        body = (page.extract_text() or "").strip()
        if body:
            parts.append(f"# Sayfa {i}\n\n{body}")
    return clean_text("\n\n".join(parts))


def docx_to_text(path: Path) -> str:
    doc = DocxDocument(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        style = p.style.name if p.style is not None else ""
        m = re.match(r"(?i)heading (\d)", style)
        if m and p.text.strip():
            parts.append(f"{'#' * min(int(m.group(1)), 3)} {p.text.strip()}")
        else:
            parts.append(p.text)
    for table in doc.tables:
        parts.append("\n".join(" | ".join(c.text.strip() for c in row.cells)
                               for row in table.rows))
    return clean_text("\n\n".join(parts))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return re.sub(r"\n{3,}", "\n\n", read_text(path)).strip()
    if suffix == ".html":
        return html_to_text(read_text(path))
    if suffix == ".pdf":
        return pdf_to_text(path)
    if suffix == ".docx":
        return docx_to_text(path)
    raise ValueError(f"desteklenmeyen tür: {suffix}")


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def load_corpus(root: Path) -> tuple[list[Doc], list[str]]:
    docs: list[Doc] = []
    report: list[str] = []
    seen: dict[str, str] = {}
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = load_file(path)
        except Exception as e:  # tek dosya tüm indekslemeyi durdurmasın
            report.append(f"HATA {rel}: {type(e).__name__}: {e}")
            continue
        if not text.strip():
            report.append(f"BOŞ {rel}: çıkarılabilir metin yok")
            continue
        h = content_hash(text)
        if h in seen:
            report.append(f"KOPYA {rel} = {seen[h]}")
            continue
        seen[h] = rel
        docs.append(Doc(rel, path.suffix.lower().lstrip("."), text, len(text)))
    return docs, report


def run(root: Path = config.SOURCE_ROOT, data_dir: Path = config.DATA_DIR) -> list[Doc]:
    if not root.is_dir():
        raise SystemExit(f"kaynak klasör yok: {root}")
    docs, report = load_corpus(root)
    write_jsonl(data_dir / "docs.jsonl", [asdict(d) for d in docs])
    (data_dir / "load_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"load: {len(docs)} doküman, {len(report)} rapor satırı → {data_dir / 'docs.jsonl'}")
    return docs
