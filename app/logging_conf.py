"""Structured logging with request correlation.

Two hard rules for a moderation service:

1. **Every log line carries the request id** returned to the client, so a user
   complaint ("my photo was rejected") can be traced end to end.
2. **Images are never logged** - not their bytes, not their content, and URLs are
   logged without query strings, which is where signed tokens live.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from flask import g, has_request_context

REQUEST_ID_ATTRIBUTE = "request_id"

#: LogRecord attributes that are part of the standard machinery, not our payload.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


def redact_url(url: str | None) -> str | None:
    """Strip the query string and any credentials from a URL before logging it."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.username:
            host = f"<redacted-user>@{host}"
        netloc = host if not parts.port else f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:  # pragma: no cover - defensive
        return "<url-inválida>"


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every record emitted in a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = "-"
        if has_request_context():
            request_id = getattr(g, REQUEST_ID_ATTRIBUTE, None) or "-"
        setattr(record, REQUEST_ID_ATTRIBUTE, request_id)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line - what log aggregators actually want."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, REQUEST_ID_ATTRIBUTE, "-"),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    """Human-readable single-line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, REQUEST_ID_ATTRIBUTE, "-")
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_") and key != REQUEST_ID_ATTRIBUTE
        }
        suffix = ""
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            suffix = f" | {rendered}"
        base = f"{record.levelname:<5} [{request_id}] {record.name}: {record.getMessage()}{suffix}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def setup_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    stream: Any = None,
) -> None:
    """Configure the root logger once, at application start-up."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else PlainFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Third-party libraries are chatty at INFO and would drown moderation events.
    for noisy in ("werkzeug", "urllib3", "PIL", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit ``event`` with structured fields attached to the record."""
    logger.log(level, event, extra={k: v for k, v in fields.items() if k not in _RESERVED})
