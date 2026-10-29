"""Embedding providers must turn *every* malformed response into EmbeddingError (so retrieval degrades to keyword search instead of a 500)."""
import numpy as np
import pytest

from app.embeddings import EmbeddingError, HFInferenceEmbedder, OpenAICompatEmbedder


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("bad", request=None, response=None)


class _Client:
    def __init__(self, body, status=200):
        self.body, self.status, self.calls = body, status, []

    def post(self, url, headers=None, json=None):
        self.calls.append((url, json))
        return _Resp(self.body, self.status)


def _hf(body, status=200, dim=3):
    return HFInferenceEmbedder("https://router.example/hf-inference/", "k", "org/model", dim, client=_Client(body, status), sleep=lambda s: None)


def test_hf_route_url_payload_and_parsing():
    e = _hf([[3.0, 4.0, 0.0]])
    v = e.embed_query("hello")
    assert e.client.calls[0] == ("https://router.example/hf-inference/models/org/model", {"inputs": ["hello"]})
    assert np.allclose(v, [0.6, 0.8, 0.0]) and e.name == "hf:org/model"
    assert _hf([1.0, 0.0, 0.0]).embed_query("x").shape == (3,)  # a bare vector for a single input is accepted


def test_hf_error_dict_wrong_dimension_and_token_level_output_are_embedding_errors():
    for body in ({"error": "Model is currently loading"}, [[1.0, 2.0]], [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]], "oops", None):
        with pytest.raises(EmbeddingError):
            _hf(body).embed_query("x")


def test_openai_shape_list_body_no_longer_escapes_as_typeerror():
    e = OpenAICompatEmbedder("http://x/v1", "k", "m", 3, client=_Client([[1.0, 2.0, 3.0]]), sleep=lambda s: None)  # a list where a dict is expected
    with pytest.raises(EmbeddingError):
        e.embed_query("x")
    ok = OpenAICompatEmbedder("http://x/v1", "k", "m", 2, client=_Client({"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}))
    assert np.allclose(ok.embed_documents(["a", "b"]), [[1, 0], [0, 1]])


def test_http_error_status_is_an_embedding_error():
    with pytest.raises(EmbeddingError):
        _hf({}, status=404).embed_query("x")


def test_malformed_base_url_is_a_clear_embedding_error_not_an_unhandled_exception():
    for bad in ("[https://router.example/hf-inference](https://router.example/hf-inference)", "router.example", "", '"https://x"', "https://x y"):
        e = HFInferenceEmbedder(bad, "k", "org/model", 3, client=_Client([[1.0, 2.0, 3.0]]), sleep=lambda s: None)
        with pytest.raises(EmbeddingError, match="EMBEDDING_BASE_URL"):
            e.embed_query("x")
        assert not e.client.calls  # failed before any request was made
    HFInferenceEmbedder("https://router.example/hf-inference", "k", "m", 3, client=_Client([[1.0, 2.0, 3.0]])).embed_query("x")
