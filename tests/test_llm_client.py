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
    assert llm.chat([]).content == "done" and calls["n"] == 3 and sleeps == [2.5, 2.5]


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


def test_404_model_not_found_is_a_config_error_and_not_retried():
    from app.llm.base import LLMConfigError
    calls = {"n": 0}
    def h(req):
        calls["n"] += 1
        return httpx.Response(404, json={"error": {"code": "model_not_found"}})
    llm, sleeps = make(h, retries=3)
    with pytest.raises(LLMConfigError) as e:
        llm.chat([])
    assert calls["n"] == 1 and sleeps == [] and "LLM_MODEL" in str(e.value)


def test_output_token_floor_reasoning_effort_and_fast_tool_retry():
    seen, sleeps, calls = [], [], {"n": 0}
    def h(req):
        calls["n"] += 1
        seen.append(json.loads(req.content))
        if calls["n"] == 1:
            return httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "Failed to parse tool call arguments as JSON"}})
        return httpx.Response(200, json=ok_body(content="x"))
    llm = OpenAICompatLLM("http://x/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(h)), sleep=sleeps.append,
                          min_output_tokens=2048, reasoning_effort="low")
    llm.chat([], max_tokens=300)
    assert seen[0]["max_tokens"] == 2048 and seen[0]["reasoning_effort"] == "low" and sleeps == [0.3]
    plain = OpenAICompatLLM("http://x/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(h)))
    seen.clear(); plain.chat([], max_tokens=5000)
    assert seen[0]["max_tokens"] == 5000 and "reasoning_effort" not in seen[0]


def test_rejected_tool_call_with_invented_tool_name_is_salvaged():
    answer = {"status": "answered", "answer": "SEV2", "claims": [{"text": "Severity: SEV2", "evidence_ids": ["E1"]}]}
    failed = json.dumps({"name": "json", "arguments": answer})
    calls = {"n": 0}
    def h(req):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "attempted to call tool 'json' which was not in request.tools", "failed_generation": failed}})
    llm, sleeps = make(h)
    tools = [{"type": "function", "function": {"name": "submit_answer"}}]
    r = llm.chat([], tools=tools, tool_choice={"type": "function", "function": {"name": "submit_answer"}})
    assert calls["n"] == 1 and sleeps == [] and r.tool_calls[0].name == "submit_answer" and r.tool_calls[0].arguments["status"] == "answered"
    r2 = llm.chat([], tools=tools + [{"type": "function", "function": {"name": "submit_plan"}}], tool_choice="auto")  # inferred from the argument keys
    assert r2.tool_calls[0].name == "submit_answer"


def test_truncated_failed_generation_is_not_salvaged_and_is_retried():
    def h(req):
        return httpx.Response(400, json={"error": {"code": "tool_use_failed", "failed_generation": '{"name": "json", "arguments": {"answer": "cut off'}})
    llm, sleeps = make(h, retries=1)
    with pytest.raises(LLMBadResponse):
        llm.chat([], tools=[{"type": "function", "function": {"name": "submit_answer"}}])
    assert sleeps == [0.3]


def test_client_side_pacing_between_calls():
    sleeps = []
    llm = OpenAICompatLLM("http://x/v1", "k", "m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body(content="x")))),
                          sleep=sleeps.append, min_interval_s=5.0)
    llm.chat([]); llm.chat([])
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 5.0


def test_429_waits_the_requested_time_and_quota_errors_are_not_retried():
    from app.llm.base import LLMQuotaExhausted
    from app.llm.openai_compat import parse_wait_seconds
    assert parse_wait_seconds(httpx.Headers({}), "Please try again in 6m31.5s.") == 391.5
    assert parse_wait_seconds(httpx.Headers({}), "Please try again in 850ms") == 0.85
    assert parse_wait_seconds(httpx.Headers({"retry-after": "12"}), "") == 12.0
    calls = {"n": 0}
    def short(req):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "Rate limit reached (TPM). Please try again in 7.5s"}}) if calls["n"] == 1 else httpx.Response(200, json=ok_body(content="ok"))
    llm, sleeps = make(short)
    assert llm.chat([]).content == "ok" and sleeps == [8.0]
    def daily(req): return httpx.Response(429, json={"error": {"message": "tokens per day (TPD) limit. Please try again in 20m3s"}})
    llm, sleeps = make(daily)
    with pytest.raises(LLMQuotaExhausted) as e:
        llm.chat([])
    assert sleeps == [] and "20 min" in str(e.value)


def test_strict_build_llm_never_falls_back_silently():
    from app.container import build_llm
    from tests.conftest import make_settings
    with pytest.raises(ValueError):
        build_llm(make_settings(llm_provider="groq", llm_api_key="k"), strict=True)  # typo / unknown provider
    with pytest.raises(ValueError):
        build_llm(make_settings(llm_provider="openai", llm_api_key=None), strict=True)
    with pytest.raises(ValueError):
        build_llm(make_settings(llm_provider="openai", llm_api_key="k", llm_base_url="nope"), strict=True)
    assert build_llm(make_settings(llm_provider="heuristic"), strict=True).name == "heuristic-extractive"
    assert build_llm(make_settings(llm_provider="groq"), strict=False).name == "heuristic-extractive"  # the app itself stays up
