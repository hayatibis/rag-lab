"""Tüm sabitler tek yerde; env ile ezilebilenler burada okunur."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAG_DATA", PROJECT_ROOT / "data"))
SOURCE_ROOT = Path(os.environ.get("RAG_SOURCE", PROJECT_ROOT / "sample_docs"))

INCLUDE_SUFFIXES = {".md", ".html", ".pdf", ".docx"}
EXCLUDE_DIRS = (".git", "archive/old")
EXCLUDE_NAME_SUFFIXES = ("-offline.html",)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"
CLAUDE_MODEL = "claude-sonnet-5"

CHUNK_TOKENS = 800
OVERLAP_TOKENS = 120
CHARS_PER_TOKEN = 4
TOP_K = 8
