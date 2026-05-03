"""
PII redaction helpers (extension E5).

Used in log records and (optionally) in agent input/output. The patterns
are conservative — false-positives are fine, leakage is not.
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Long hex tokens that look like API keys / build SHAs (>=24 hex chars).
TOKEN_RE = re.compile(r"\b[A-Fa-f0-9]{24,}\b")
# Bearer / API key headers.
BEARER_RE = re.compile(r"(?i)(authorization|api[_-]?key)\s*[:=]\s*\S+")


def redact(text: str) -> str:
    """Best-effort PII / secret redaction for log lines."""
    if not text:
        return text
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = BEARER_RE.sub("[REDACTED_AUTH]", text)
    return text


def redact_processor(_logger: object, _method: str, event_dict: dict) -> dict:
    """structlog processor: redact every string value in the event dict."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict
