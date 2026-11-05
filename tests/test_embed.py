import json

import httpx
import numpy as np
import pytest

from rag.embed import OllamaError, embed_texts, normalize_rows


def mock_client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))


def test_normalize_rows_unit_length_and_zero_safe():
    m = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert m.dtype == np.float32
    assert np.allclose(m[0], [0.6, 0.8])
    assert np.allclose(m[1], [0.0, 0.0])


def test_embed_texts_batches_and_normalizes():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        assert body["model"] == "bge-m3"
        return httpx.Response(200, json={"embeddings": [[2.0, 0.0]] * len(body["input"])})

    v = embed_texts(["a", "b", "c"], batch_size=2, client=mock_client(handler))
    assert calls == [2, 1]
    assert v.shape == (3, 2)
    assert np.allclose(v[:, 0], 1.0)


def test_embed_texts_explains_missing_ollama():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        embed_texts(["a"], client=mock_client(handler))


def test_embed_texts_explains_missing_model():
    def handler(request):
        return httpx.Response(404, json={"error": 'model "bge-m3" not found'})

    with pytest.raises(OllamaError, match="ollama pull bge-m3"):
        embed_texts(["a"], client=mock_client(handler))
