"""5. adım: top-k parçayı prompt'a koyup Claude'dan kaynaklı cevap alır."""
from __future__ import annotations

import os

import anthropic
import httpx

from rag import config
from rag.search import Hit

# Geçici dış hatalar (Claude 529/rate limit/bağlantı, Ollama timeout/5xx): tek soruyu düşürür, oturumu değil
TRANSIENT_ERRORS = (anthropic.APIError, httpx.HTTPError)

SYSTEM_PROMPT = """Sen bir doküman asistanısın. Kullanıcının sorusunu YALNIZCA verilen numaralı doküman parçalarına dayanarak cevapla.

Kurallar:
- Her iddianın sonuna dayandığın parçanın numarasını [n] biçiminde ekle.
- Parçalarda cevap yoksa ya da yetersizse tahmin yürütme; "Dokümanlarda bulamadım." de ve neyin eksik olduğunu söyle.
- Parçalar birbiriyle çelişiyorsa bunu açıkça belirt ve iki tarafı da atıfla ver.
- Türkçe cevap ver."""


class MissingApiKey(RuntimeError):
    pass


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f'<parca no="{n}">\n{h.text}\n</parca>' for n, h in enumerate(hits, 1)]
    return "Doküman parçaları:\n\n" + "\n\n".join(blocks) + f"\n\nSoru: {question}"


def answer(question: str, hits: list[Hit], client=None) -> str:
    return generate(question, hits, client)[0]


def generate(question: str, hits: list[Hit], client=None) -> tuple[str, dict]:
    """Cevap metni + token kullanımı ({input_tokens, output_tokens})."""
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingApiKey("ANTHROPIC_API_KEY tanımlı değil — "
                                "`export ANTHROPIC_API_KEY=...` ya da `--retrieve-only` kullan")
        client = anthropic.Anthropic()
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = getattr(resp, "usage", None)
    usage = {"input_tokens": getattr(u, "input_tokens", None),
             "output_tokens": getattr(u, "output_tokens", None),
             "stop_reason": getattr(resp, "stop_reason", None)}  # "max_tokens" ⇒ cevap kesildi
    return text, usage
