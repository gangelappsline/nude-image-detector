"""Cross-cutting request protection: authentication, client identity, rate limits.

Deliberately dependency-free: the primitives here are per-process.  If you run
several replicas, put the real limits in your gateway or swap :class:`RateLimiter`
for a Redis-backed one - the call sites do not change.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from flask import Request, request

from .config import Settings
from .errors import RateLimited, Unauthorized

API_KEY_HEADER = "X-API-Key"
_API_KEY_QUERY_ARG = "api_key"


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def verify_api_key(candidate: str | None, settings: Settings) -> bool:
    """Constant-time check against the configured key list."""
    if not settings.api_keys:
        return True
    if not candidate:
        return False
    encoded = candidate.encode("utf-8")
    return any(hmac.compare_digest(encoded, key.encode("utf-8")) for key in settings.api_keys)


def extract_api_key(req: Request) -> str | None:
    header = req.headers.get(API_KEY_HEADER)
    if header:
        return header.strip()
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # Query strings end up in logs and browser history; accepted only for
    # convenience (e.g. an <img> tag) and documented as discouraged.
    query_value = req.args.get(_API_KEY_QUERY_ARG)
    return query_value.strip() if query_value else None


def require_auth(req: Request, settings: Settings) -> None:
    if not settings.api_keys:
        return
    if not verify_api_key(extract_api_key(req), settings):
        raise Unauthorized(
            "Falta o es inválida la clave de API.",
            details={"header": API_KEY_HEADER},
        )


# --------------------------------------------------------------------------- #
# Client identity
# --------------------------------------------------------------------------- #
def client_ip(req: Request, settings: Settings) -> str:
    """Best-effort client address, trusting at most ``trusted_proxy_count`` hops.

    ``X-Forwarded-For`` is attacker-controlled unless it comes from your own
    proxy, so the right-most N entries are used (N = number of proxies you
    actually run) instead of the left-most one.
    """
    remote = req.remote_addr or "unknown"
    if settings.trusted_proxy_count <= 0:
        return remote

    forwarded = req.headers.get("X-Forwarded-For")
    if not forwarded:
        return remote

    candidates = [part.strip() for part in forwarded.split(",") if part.strip()]
    if not candidates:
        return remote
    index = max(0, len(candidates) - settings.trusted_proxy_count)
    return candidates[index]


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int
    key: str

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.retry_after),
        }


class RateLimiter:
    """Fixed-window counter per identity (API key when present, else IP)."""

    def __init__(self, settings: Settings, *, clock: Callable = time.monotonic) -> None:
        self.enabled = settings.rate_limit_enabled
        self.limit = settings.rate_limit_requests
        self.window = max(1, settings.rate_limit_window_seconds)
        self._clock = clock
        self._hits: dict[str, list[int]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, identity: str, *, cost: int = 1) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(True, self.limit, self.limit, 0, identity)

        now = int(self._clock())
        window_start = now - self.window
        with self._lock:
            timestamps = [t for t in self._hits[identity] if t > window_start]
            if len(timestamps) + cost > self.limit:
                retry_after = max(1, (timestamps[0] + self.window) - now) if timestamps else 1
                self._hits[identity] = timestamps
                return RateLimitDecision(False, self.limit, 0, retry_after, identity)
            timestamps.extend([now] * max(1, cost))
            self._hits[identity] = timestamps
            remaining = max(0, self.limit - len(timestamps))
            return RateLimitDecision(True, self.limit, remaining, self.window, identity)

    def enforce(self, identity: str, *, cost: int = 1) -> RateLimitDecision:
        decision = self.check(identity, cost=cost)
        if not decision.allowed:
            raise RateLimited(
                f"Límite de {self.limit} solicitudes por {self.window}s superado.",
                details={"identity_type": "api_key" if identity.startswith("key:") else "ip"},
                retry_after=decision.retry_after,
            )
        return decision

    def prune(self) -> int:
        """Drop stale buckets; returns how many identities were forgotten."""
        now = int(self._clock())
        window_start = now - self.window
        removed = 0
        with self._lock:
            for identity in list(self._hits):
                kept = [t for t in self._hits[identity] if t > window_start]
                if kept:
                    self._hits[identity] = kept
                else:
                    del self._hits[identity]
                    removed += 1
        return removed

    def size(self) -> int:
        with self._lock:
            return len(self._hits)


def identity_for(req: Request, settings: Settings) -> str:
    """Stable bucket key: the API key when authenticated, otherwise the IP."""
    key = extract_api_key(req)
    if key and settings.api_keys and verify_api_key(key, settings):
        # Hash it: rate-limit state should not keep raw credentials in memory.
        digest = hmac.new(b"nid-ratelimit", key.encode(), "sha256").hexdigest()[:16]
        return f"key:{digest}"
    return f"ip:{client_ip(req, settings)}"


def enforce_request_limits(req: Request, settings: Settings, limiter: RateLimiter, *, cost: int = 1) -> RateLimitDecision:
    return limiter.enforce(identity_for(req, settings), cost=cost)


def request_cost(req: Request) -> int:
    """Charge batch requests proportionally to the number of images."""
    files = list(_iter_uploads(req))
    if files:
        return max(1, len(files))
    payload = req.get_json(silent=True)
    if isinstance(payload, dict):
        urls = payload.get("urls")
        if isinstance(urls, Iterable) and not isinstance(urls, (str, bytes)):
            return max(1, len(list(urls)))
        if payload.get("url"):
            return 1
    return 1


def _iter_uploads(req: Request) -> Iterable[str]:
    try:
        return list(req.files.keys())
    except Exception:  # pragma: no cover - not multipart
        return []


def current_identity() -> str:
    """Convenience wrapper for blueprints (uses the ambient Flask request)."""
    from flask import current_app

    settings: Settings = current_app.config["NID_SETTINGS"]
    return identity_for(request, settings)
