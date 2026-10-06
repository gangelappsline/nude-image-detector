"""Tiny thread-safe TTL/LRU cache for analysis results.

People re-upload the same avatar, the same sticker or the same offending photo
over and over.  Keying on the image SHA-256 lets us answer duplicates without
touching the model - which is usually the most expensive part of the request.

It is an in-process cache on purpose: no external dependency, and a false miss
only costs one inference.  For a multi-worker deployment set a bigger
``NID_CACHE_MAXSIZE`` or put a shared cache in front of the API.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
        }


class TTLCache:
    """LRU cache whose entries expire after ``ttl`` seconds."""

    def __init__(
        self,
        maxsize: int = 512,
        ttl: float = 3600.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        enabled: bool = True,
    ) -> None:
        self.maxsize = max(0, int(maxsize))
        self.ttl = float(ttl)
        self._clock = clock
        self.enabled = enabled and self.maxsize > 0 and self.ttl > 0
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._data[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            self._data.move_to_end(key)
            self._stats.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        expires_at = self._clock() + self.ttl
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while self.maxsize and len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self._stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.size = len(self._data)
            snapshot = CacheStats(**self._stats.__dict__)
        return snapshot
