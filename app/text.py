"""Text utilities shared by BM25, the reranker, the verifier and de-duplication.
Everything here is our own implementation (tokenizer, light stemmer, SimHash)."""
from __future__ import annotations

import re
import zlib
from collections import Counter

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those which who whom whose what when where why how did do does done has have had having will would
can could should may might shall not no nor than then so such into over under about after before during
their there them they he she his her you your we our us i me my also any all each both more most other
some only own same too very just per via
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-_.]")
_ID_RE = re.compile(r"\b[A-Za-z]{2,6}-\d{2,6}\b")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007\u2008\u200a\u3000"), " ")
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def normalize_typography(s: str) -> str:
    """Map typographic dashes / non-breaking spaces to ASCII and drop zero-width characters.
    LLMs often write 'CHG\u20110025' (non-breaking hyphen); without this identifiers silently stop matching."""
    return s.translate({**_DASHES, **_SPACES, **_INVISIBLE})


def stem(tok: str) -> str:
    """Very light suffix stripper (only for purely alphabetic tokens). Also drops a trailing 'e'
    so that cause / caused / causes / causing all map to the same stem."""
    if not tok.isalpha() or len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        tok = tok[:-2]
    elif tok.endswith("ing") and len(tok) > 5:
        tok = tok[:-3]
    elif tok.endswith("ed") and len(tok) > 4:
        tok = tok[:-2]
    elif tok.endswith("es") and len(tok) > 4:
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    if len(tok) > 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Lowercase tokens. Compound tokens such as 'inc-0012' or 'payments-api' are emitted whole AND
    split into parts, so both exact identifier matches and partial matches work."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        raw = m.group(0)
        parts = [p for p in _SPLIT_RE.split(raw) if p]
        if len(parts) > 1:
            out.append(raw)
            for p in parts:
                p = stem(p)
                if len(p) >= 2 and (keep_stop or p not in STOPWORDS):
                    out.append(p)
        else:
            t = stem(raw)
            if keep_stop or t not in STOPWORDS:
                out.append(t)
    return out


def identifiers(text: str) -> list[str]:
    """Identifier-like tokens (INC-0012, CHG-0123, JIRA-77) lower-cased, order preserved, unique."""
    seen: dict[str, None] = {}
    for m in _ID_RE.finditer(text):
        seen.setdefault(m.group(0).lower(), None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]


def count_tokens(text: str) -> int:
    """Cheap whitespace token estimate (used for chunk sizing and budget accounting)."""
    return len(text.split())


# --------------------------------------------------------------------------- SimHash
def _h64(token: str) -> int:
    b = token.encode("utf-8")
    return (zlib.crc32(b) << 32) | zlib.crc32(b, 0x9E3779B1)


def simhash(text: str) -> int:
    """64-bit SimHash over unigrams + bigrams. Near-identical texts differ in only a few bits."""
    toks = tokenize(text)
    if not toks:
        return 0
    feats = Counter(toks)
    feats.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    v = [0] * 64
    for f, w in feats.items():
        h = _h64(f)
        for i in range(64):
            v[i] += w if (h >> i) & 1 else -w
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simhash_hex(text: str) -> str:
    return f"{simhash(text):016x}"
