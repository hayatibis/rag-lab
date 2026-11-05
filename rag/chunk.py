"""2. adım: doküman metnini başlık-duyarlı parçalara böler → data/chunks.jsonl."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rag import config
from rag.jsonl import read_jsonl, write_jsonl

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


@dataclass
class Section:
    heading_path: list[str]
    body: str


@dataclass
class Chunk:
    id: str
    path: str
    heading_path: list[str]
    text: str
    n_tokens_est: int


def _flush(sections: list[Section], path: list[str], buf: list[str]) -> None:
    body = "\n".join(buf).strip()
    if body:
        sections.append(Section(list(path), body))


def split_sections(text: str) -> list[Section]:
    """Başlık (#, ##, ###) sınırlarından böler; kod bloklarındaki '#' başlık sayılmaz."""
    sections: list[Section] = []
    stack: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            _flush(sections, stack, buf)
            buf = []
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2)]
        buf.append(line)
    _flush(sections, stack, buf)
    return sections


def overlap_tail(text: str, n: int) -> str:
    """Metnin son ~n karakteri, kelime başından başlayacak şekilde."""
    if n <= 0:
        return ""
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def split_long(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    for para in body.split("\n\n"):
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(para[:cut])
            para = para[cut:].lstrip()
        if para:
            pieces.append(para)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        cand = f"{cur}\n\n{p}" if cur else p
        if not cur or len(cand) <= max_chars:
            cur = cand
            continue
        chunks.append(cur)
        tail = overlap_tail(cur, overlap_chars)
        cur = f"{tail}\n\n{p}" if tail else p
    chunks.append(cur)
    return chunks


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def merge_short(sections: list[Section], max_chars: int) -> list[Section]:
    """Aynı üst başlığı paylaşan ardışık kısa bölümleri sınır dolana kadar birleştirir."""
    merged: list[Section] = []
    for s in sections:
        if merged:
            prev = merged[-1]
            common = _common_prefix(prev.heading_path, s.heading_path)
            same_top = bool(common) or (not prev.heading_path and not s.heading_path)
            if same_top and len(prev.body) + 2 + len(s.body) <= max_chars:
                merged[-1] = Section(common, f"{prev.body}\n\n{s.body}")
                continue
        merged.append(s)
    return merged


def make_label(path: str, heading_path: list[str]) -> str:
    return "[" + " › ".join([path, *heading_path]) + "]"


def chunk_doc(path: str, text: str,
              max_tokens: int = config.CHUNK_TOKENS,
              overlap_tokens: int = config.OVERLAP_TOKENS) -> list[Chunk]:
    max_chars = max_tokens * config.CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * config.CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    for s in merge_short(split_sections(text), max_chars):
        for piece in split_long(s.body, max_chars, overlap_chars):
            full = f"{make_label(path, s.heading_path)}\n{piece}"
            chunks.append(Chunk(
                id=f"{path}#{len(chunks)}",
                path=path,
                heading_path=s.heading_path,
                text=full,
                n_tokens_est=len(full) // config.CHARS_PER_TOKEN,
            ))
    return chunks


def run(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    docs = read_jsonl(data_dir / "docs.jsonl")
    chunks = [c for d in docs for c in chunk_doc(d["path"], d["text"])]
    write_jsonl(data_dir / "chunks.jsonl", [asdict(c) for c in chunks])
    sizes = [c.n_tokens_est for c in chunks] or [0]
    print(f"chunk: {len(docs)} doküman → {len(chunks)} parça "
          f"(ort. {sum(sizes) // len(sizes)} token, en büyük {max(sizes)}) "
          f"→ {data_dir / 'chunks.jsonl'}")
    return chunks
