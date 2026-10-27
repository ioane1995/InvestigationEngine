"""Thin OpenAI-compatible chat client (Groq, Gemini's OpenAI endpoint, OpenRouter, OpenAI, vLLM...).
Written directly on httpx so retries, timeouts and error mapping are explicit and unit-testable."""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, Optional

import httpx

from app.llm.base import (LLMAuthError, LLMBadResponse, LLMConfigError, LLMError, LLMRateLimited, LLMResponse, LLMTimeout,
                          ToolCall, Usage)

log = logging.getLogger(__name__)


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: Optional[str], model: str, timeout: float = 45.0, max_retries: int = 3,
                 client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = time.sleep):
        base_url = (base_url or "").strip().strip("'\"")
        if not base_url.lower().startswith(("http://", "https://")):
            raise LLMConfigError(f"LLM_BASE_URL must start with http:// or https:// (got {base_url!r}); check your .env / environment variables")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.name = model, f"openai-compat:{model}"
        self.max_retries, self._sleep = max_retries, sleep
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None, tool_choice: Any = None,
             temperature: float = 0.0, max_tokens: int = 1500) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
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

    def _once(self, body: dict[str, Any]) -> LLMResponse:
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
            e = LLMRateLimited("rate limited (429)")
            ra = r.headers.get("retry-after")
            e.retry_after = min(float(ra), 30.0) if ra and ra.replace(".", "", 1).isdigit() else None  # type: ignore[attr-defined]
            raise e
        if r.status_code >= 500:
            raise LLMBadResponse(f"provider error {r.status_code}")
        if r.status_code >= 400:
            # e.g. 400 'tool_use_failed' when the model emitted a malformed tool call: worth one more try
            raise LLMBadResponse(f"bad request {r.status_code}: {r.text[:300]}")
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
