"""Thin OpenAI-compatible chat client (Groq, Gemini's OpenAI endpoint, OpenRouter, OpenAI, vLLM...).
Written directly on httpx so retries, timeouts and error mapping are explicit and unit-testable."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, Optional

import httpx

from app.llm.base import (LLMAuthError, LLMBadResponse, LLMConfigError, LLMError, LLMQuotaExhausted, LLMRateLimited, LLMResponse, LLMTimeout,
                          ToolCall, Usage)

log = logging.getLogger(__name__)


_DUR = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
MAX_WAIT_S = 90.0  # a provider asking for longer than this is telling us about a quota, not a burst


def parse_wait_seconds(headers: "httpx.Headers", body: str) -> Optional[float]:
    """Seconds the provider asks us to wait: the Retry-After header, or text such as 'Please try again in 6m31.2s'."""
    ra = headers.get("retry-after")
    if ra and ra.replace(".", "", 1).isdigit():
        return float(ra)
    m = re.search(r"try again in ([0-9hms.]+)", body)
    if m:
        unit = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        parts = _DUR.findall(m.group(1))
        if parts:
            return sum(float(n) * unit[u] for n, u in parts)
    return None


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: Optional[str], model: str, timeout: float = 45.0, max_retries: int = 3,
                 client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = time.sleep,
                 min_output_tokens: int = 2048, reasoning_effort: Optional[str] = None, min_interval_s: float = 0.0):
        base_url = (base_url or "").strip().strip("'\"")
        if not base_url.lower().startswith(("http://", "https://")):
            raise LLMConfigError(f"LLM_BASE_URL must start with http:// or https:// (got {base_url!r}); check your .env / environment variables")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.name = model, f"openai-compat:{model}"
        self.max_retries, self._sleep = max_retries, sleep
        self.min_output_tokens, self.reasoning_effort = min_output_tokens, reasoning_effort or None
        self.min_interval_s, self._last_call = min_interval_s, 0.0  # optional client-side pacing for free-tier rate limits
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None, tool_choice: Any = None,
             temperature: float = 0.0, max_tokens: int = 1500) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": max(max_tokens, self.min_output_tokens)}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        last: Optional[LLMError] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._once(body)
            except LLMError as e:
                last = e
                if not e.retryable or attempt == self.max_retries:
                    raise
                delay = getattr(e, "retry_after", None) or min(20.0, 1.0 * 2 ** attempt) + random.random() * 0.4
                log.warning("LLM call failed (%s); retry %d/%d in %.1fs", e, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
        raise last or LLMError("unreachable")

    @staticmethod
    def _salvage(body: dict[str, Any], r: httpx.Response) -> Optional[LLMResponse]:
        """Some providers reject a generation that is *almost* a valid tool call (e.g. the model invents a tool called
        'json' but puts the correct arguments in it) and return the rejected text as `failed_generation`.
        If it parses as JSON and we can tell which of OUR tools it was meant for, use it instead of wasting a retry."""
        try:
            gen = r.json()["error"]["failed_generation"]
            obj = json.loads(gen) if isinstance(gen, str) else gen
            args = obj.get("arguments", obj) if isinstance(obj, dict) else None
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                return None
        except (ValueError, KeyError, TypeError, AttributeError):
            return None
        names = [t["function"]["name"] for t in body.get("tools", [])]
        choice = body.get("tool_choice")
        target = choice["function"]["name"] if isinstance(choice, dict) else None
        if target is None:
            if {"status", "answer"} <= args.keys() and "submit_answer" in names:
                target = "submit_answer"
            elif "sub_queries" in args and "submit_plan" in names:
                target = "submit_plan"
            elif isinstance(obj, dict) and obj.get("name") in names:
                target = obj["name"]
        if target is None or target not in names:
            return None
        log.warning("salvaged a rejected tool call as %s", target)
        return LLMResponse(None, [ToolCall("salvaged", target, args)], Usage(), body.get("model", ""))

    def _once(self, body: dict[str, Any]) -> LLMResponse:
        if self.min_interval_s > 0:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
        try:
            r = self.client.post(self.url, headers=self.headers, json=body)
        except (httpx.InvalidURL, httpx.UnsupportedProtocol) as e:
            raise LLMConfigError(f"invalid LLM_BASE_URL: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"timeout: {e}") from e
        except httpx.HTTPError as e:
            t = LLMTimeout(f"network error: {e}")
            raise t from e
        if r.status_code in (401, 403):
            raise LLMAuthError(f"authentication failed ({r.status_code}) - check LLM_API_KEY")
        if r.status_code == 429:
            wait = parse_wait_seconds(r.headers, r.text)
            detail = r.text[:240].replace("\n", " ")
            if wait is not None and wait > MAX_WAIT_S:
                raise LLMQuotaExhausted(f"quota exhausted: provider asks to wait {wait / 60:.0f} min ({detail})")
            e = LLMRateLimited(f"rate limited (429): {detail}")
            e.retry_after = min(wait + 0.5, MAX_WAIT_S) if wait is not None else None  # type: ignore[attr-defined]
            raise e
        if r.status_code >= 500:
            raise LLMBadResponse(f"provider error {r.status_code}")
        if r.status_code == 404:
            raise LLMConfigError(f"model or endpoint not found (404). Check LLM_MODEL={self.model!r} and LLM_BASE_URL. Provider said: {r.text[:200]}")
        if r.status_code >= 400:
            # e.g. 400 'tool_use_failed' when the model emitted a malformed tool call: worth one more try
            if "tool_use_failed" in r.text:
                salvaged = self._salvage(body, r)
                if salvaged is not None:
                    return salvaged
            e = LLMBadResponse(f"bad request {r.status_code}: {r.text[:300]}")
            if "tool_use_failed" in r.text:
                e.retry_after = 0.3  # type: ignore[attr-defined]  # a malformed generation is not a rate limit: retry almost immediately
            raise e
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMBadResponse(f"unparseable response: {r.text[:200]}") from e
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args, err = json.loads(raw), None
                if not isinstance(args, dict):
                    args, err = {}, "tool arguments are not a JSON object"
            except json.JSONDecodeError as e:
                args, err = {}, f"invalid JSON in tool arguments: {e}"
            calls.append(ToolCall(tc.get("id") or f"call_{len(calls)}", tc.get("function", {}).get("name", ""), args, err))
        u = data.get("usage") or {}
        return LLMResponse(msg.get("content"), calls, Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))),
                           data.get("model", self.model))
