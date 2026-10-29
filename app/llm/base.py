"""LLM abstraction: one `chat()` method with tool calling, typed errors and token accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class LLMError(RuntimeError):
    """Base class. `retryable` tells callers whether trying again can help."""
    retryable = False


class LLMRateLimited(LLMError):
    retryable = True


class LLMQuotaExhausted(LLMError):
    """The provider asks us to wait minutes/hours (daily or monthly quota). Retrying now is pointless."""
    retryable = False


class LLMTimeout(LLMError):
    retryable = True


class LLMBadResponse(LLMError):
    """Provider answered, but not in a usable shape (malformed tool call JSON, empty choices...)."""
    retryable = True


class LLMAuthError(LLMError):
    retryable = False


class LLMConfigError(LLMError):
    """Misconfiguration (bad base URL, ...). Retrying can never help, so it fails immediately."""
    retryable = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: Optional[str] = None  # set when the arguments were not valid JSON


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None,
             tool_choice: Any = None, temperature: float = 0.0, max_tokens: int = 1500) -> LLMResponse: ...
