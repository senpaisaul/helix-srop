"""
Structured logging setup with PII redaction.

All log lines include session_id, trace_id, user_id when bound via
contextvars from the request handler.
"""
from __future__ import annotations

import logging
import sys

import structlog

from app.obs.redact import redact_processor


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_processor,  # E5: PII redaction
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )
