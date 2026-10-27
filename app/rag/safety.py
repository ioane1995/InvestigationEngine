"""Prompt-injection hygiene for retrieved text. Retrieved documents are *untrusted data*:
we (1) flag instruction-like patterns, (2) neutralise tag break-outs, and (3) tell the model in the
system prompt to never follow instructions found inside <evidence>. Claim verification (verify.py)
is the last line of defence: an injected claim that the cited evidence does not support is dropped."""
from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"(reveal|print|show|leak) (your|the) (system )?(prompt|instructions|api key|secret)",
    r"do not (cite|mention) (any )?(sources?|documents?)",
    r"(always|must) (answer|respond|reply) (with|that)\b",
    r"</?(system|assistant|evidence|tool)\b",
    r"\bBEGIN (SYSTEM|ADMIN)\b",
]
_RE = re.compile("|".join(_PATTERNS), re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_RE.search(text))


def neutralize(text: str) -> str:
    """Prevent a document from closing our <evidence> wrapper or forging new ones."""
    return re.sub(r"<\s*(/?)\s*(evidence|system|assistant|tool|question|candidate_conflicts)", r"<\\\1\2", text, flags=re.I)
