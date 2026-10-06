"""Gunicorn configuration.

Sizing rule of thumb for this workload: inference is CPU-bound and the model
weighs ~110 MB RSS *per worker*.  Prefer a few workers with several threads over
many workers, and cap concurrency inside the app with
``NID_MAX_CONCURRENT_INFERENCES``.

All values can be overridden from the environment without touching this file.
"""

from __future__ import annotations

import multiprocessing
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_host = os.environ.get("NID_HOST", "0.0.0.0")
_port = _int("NID_PORT", 8000)

bind = f"{_host}:{_port}"

# CPU-bound work: a handful of workers, each with threads for I/O overlap while
# a URL is being downloaded or a body is being read.
workers = _int("NID_GUNICORN_WORKERS", min(4, max(1, (os.cpu_count() or multiprocessing.cpu_count()))))
threads = _int("NID_GUNICORN_THREADS", 4)
worker_class = os.environ.get("NID_GUNICORN_WORKER_CLASS", "gthread")

timeout = _int("NID_GUNICORN_TIMEOUT", 60)
graceful_timeout = _int("NID_GUNICORN_GRACEFUL_TIMEOUT", 15)
keepalive = _int("NID_GUNICORN_KEEPALIVE", 5)

# Each worker loads its own ONNX session; forking *after* loading it is unsafe,
# so keep preload disabled and let every worker initialise on boot.
preload_app = False

# Recycle workers periodically to contain slow memory growth under load.
max_requests = _int("NID_GUNICORN_MAX_REQUESTS", 2000)
max_requests_jitter = _int("NID_GUNICORN_MAX_REQUESTS_JITTER", 200)

# The application emits its own structured access logs; gunicorn only reports
# worker lifecycle events.
accesslog = None
errorlog = "-"
loglevel = os.environ.get("NID_GUNICORN_LOG_LEVEL", "info").lower()
capture_output = False

proc_name = "nude-image-detector"
