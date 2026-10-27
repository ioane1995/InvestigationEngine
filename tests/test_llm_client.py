import json

import httpx
import pytest

from app.llm.base import LLMAuthError, LLMBadResponse, LLMRateLimited, LLMTimeout
from app.llm.openai_compat import OpenAICompatLLM


def make(handler, retries=2):
    sleeps = []
    llm = OpenAICompatLLM("http://x/v1", "k", "m", max_retries=retries, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append)
    return llm, sleeps


def ok_body(**msg):
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"}


def test_parses_content_tool_calls_and_usage():
    body = ok_body(content=None, tool_calls=[{"id": "1", "function": {"name": "f", "arguments": '{"a": 1}'}}])
    llm, _ = make(lambda r: httpx.Response(200, json=body))
    r = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert r.tool_calls[0].arguments == {"a": 1} and r.usage.prompt_tokens == 10


def test_invalid_tool_json_is_reported_not_raised():
    body = ok_body(tool_calls=[{"id": "1", "function": {"name": "f", "arguments": "{oops"}}])
    llm, _ = make(lambda r: httpx.Response(200, json=body))
    tc = llm.chat([]).tool_calls[0]
    assert tc.error and tc.arguments == {}


def test_retries_429_then_succeeds_with_retry_after():
    calls = {"n": 0}
    def h(req):
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "2"}) if calls["n"] < 3 else httpx.Response(200, json=ok_body(content="done"))
    llm, sleeps = make(h)
    assert llm.chat([]).content == "done" and calls["n"] == 3 and sleeps == [2.0, 2.0]


def test_gives_up_after_max_retries():
    llm, _ = make(lambda r: httpx.Response(503), retries=2)
    with pytest.raises(LLMBadResponse):
        llm.chat([])


def test_auth_error_not_retried():
    calls = {"n": 0}
    def h(req):
        calls["n"] += 1
        return httpx.Response(401)
    llm, _ = make(h)
    with pytest.raises(LLMAuthError):
        llm.chat([])
    assert calls["n"] == 1


def test_timeout_mapped():
    def h(req): raise httpx.ReadTimeout("slow")
    llm, _ = make(h, retries=1)
    with pytest.raises(LLMTimeout):
        llm.chat([])


def test_malformed_payload():
    llm, _ = make(lambda r: httpx.Response(200, json={"nope": 1}), retries=0)
    with pytest.raises(LLMBadResponse):
        llm.chat([])


def test_sends_tools_and_auth_header():
    seen = {}
    def h(req):
        seen.update(json.loads(req.content)); seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=ok_body(content="x"))
    llm, _ = make(h)
    llm.chat([{"role": "user", "content": "q"}], tools=[{"type": "function", "function": {"name": "t"}}], tool_choice="required")
    assert seen["tool_choice"] == "required" and seen["auth"] == "Bearer k" and seen["temperature"] == 0.0


def test_malformed_base_url_fails_fast_and_is_not_retried():
    from app.llm.base import LLMConfigError
    with pytest.raises(LLMConfigError):
        OpenAICompatLLM("", "k", "m")
    with pytest.raises(LLMConfigError):
        OpenAICompatLLM("api.groq.com/openai/v1", "k", "m")
    assert OpenAICompatLLM('"https://api.groq.com/openai/v1"', "k", "m").url == "https://api.groq.com/openai/v1/chat/completions"

    sleeps = []
    def h(req): raise httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol.")
    llm = OpenAICompatLLM("http://x/v1", "k", "m", max_retries=3, client=httpx.Client(transport=httpx.MockTransport(h)), sleep=sleeps.append)
    with pytest.raises(LLMConfigError):
        llm.chat([])
    assert sleeps == []  # no retry storm


def test_build_llm_falls_back_instead_of_crashing_on_bad_config():
    from app.container import build_llm
    from app.llm.heuristic import HeuristicLLM
    from tests.conftest import make_settings
    assert isinstance(build_llm(make_settings(llm_provider="openai", llm_api_key="k", llm_base_url="")), HeuristicLLM)
    assert isinstance(build_llm(make_settings(llm_provider="openai", llm_api_key=None)), HeuristicLLM)
    assert build_llm(make_settings(llm_provider="openai", llm_api_key="k", llm_base_url="https://x/v1")).name.startswith("openai-compat")
