import pytest

from rag.answer import SYSTEM_PROMPT, MissingApiKey, answer, build_prompt, generate
from rag.search import Hit


def test_build_prompt_numbers_chunks_and_ends_with_question():
    hits = [Hit("a#0", "a", [], 0.9, "[a]\nbir"), Hit("b#0", "b", [], 0.8, "[b]\niki")]
    p = build_prompt("Ne?", hits)
    assert '<parca no="1">\n[a]\nbir\n</parca>' in p
    assert '<parca no="2">\n[b]\niki\n</parca>' in p
    assert p.endswith("Soru: Ne?")


def test_answer_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="retrieve-only"):
        answer("Ne?", [])


def test_answer_sends_system_prompt_and_joins_text_blocks():
    class Block:
        def __init__(self, type_, text=""):
            self.type, self.text = type_, text

    class Messages:
        def create(self, **kw):
            self.kw = kw
            return type("Resp", (), {"content": [Block("text", "cevap "), Block("text", "[1]")]})()

    client = type("Client", (), {"messages": Messages()})()
    assert answer("Ne?", [], client=client) == "cevap [1]"
    assert client.messages.kw["system"] == SYSTEM_PROMPT
    assert client.messages.kw["messages"][0]["content"].endswith("Soru: Ne?")


def test_generate_returns_text_and_token_usage():
    class Block:
        type, text = "text", "cevap [1]"

    class Messages:
        def create(self, **kw):
            usage = type("U", (), {"input_tokens": 1200, "output_tokens": 80})()
            return type("Resp", (), {"content": [Block()], "usage": usage,
                                     "stop_reason": "max_tokens"})()

    client = type("Client", (), {"messages": Messages()})()
    text, usage = generate("Ne?", [], client=client)
    assert text == "cevap [1]"
    assert usage == {"input_tokens": 1200, "output_tokens": 80, "stop_reason": "max_tokens"}
