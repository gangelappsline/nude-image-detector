"""SSRF-hardened remote image fetching.

Accepting a URL from an anonymous client turns your API into an open proxy
unless you are careful.  An attacker will try:

* ``http://127.0.0.1:8000/admin`` or ``http://[::1]/`` (loopback services)
* ``http://169.254.169.254/latest/meta-data/iam/security-credentials/`` (cloud
  metadata -> stolen temporary credentials)
* ``http://10.0.0.5/`` or ``http://192.168.1.1/`` (internal network scanning)
* ``http://public-host/`` whose DNS answer is ``127.0.0.1`` (DNS rebinding)
* ``http://allowed-host/redirect?to=http://169.254.169.254/`` (redirect hop)
* A 5 GB file, or an endless stream (resource exhaustion)

Defences implemented here, in order:

1. Scheme/port/userinfo/host validation and allow/deny lists.
2. Pre-flight DNS resolution: **every** resolved address must be public.
3. Redirects are followed manually, re-validating each hop from scratch.
4. ``Content-Length`` cap plus a hard streaming byte cap.
5. Post-connect peer-IP verification (catches DNS rebinding): if the socket we
   actually talked to is not a public address, the body is discarded.
6. Overall wall-clock deadline across all hops.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from ..config import Settings
from ..errors import (
    BlockedDestination,
    Forbidden,
    InvalidUrl,
    PayloadTooLarge,
    UnsupportedMediaType,
    UpstreamError,
    UpstreamTimeout,
)

MAX_URL_LENGTH = 2048

#: Hostnames that always resolve to (or mean) something internal, whatever DNS says.
_ALWAYS_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Content types we refuse outright: an HTML error page is not an image.
_FORBIDDEN_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "text/plain",
        "text/xml",
        "application/xhtml+xml",
        "application/json",
        "application/xml",
        "application/javascript",
    }
)


@dataclass
class FetchResult:
    """Outcome of a successful remote fetch."""

    data: bytes
    final_url: str
    status_code: int
    content_type: str | None
    elapsed_ms: float
    resolved_ips: tuple[str, ...] = ()
    peer_ip: str | None = None
    redirects: int = 0
    hops: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        """Serialisable summary - the host only, never a signed/tokenised URL."""
        return {
            "type": "url",
            "host": _safe_host(self.final_url),
            "final_url": self.final_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "redirects": self.redirects,
            "resolved_ips": list(self.resolved_ips),
            "peer_ip": self.peer_ip,
            "bytes": len(self.data),
            "fetch_ms": round(self.elapsed_ms, 1),
        }


def _safe_host(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except ValueError:  # pragma: no cover - defensive
        return None


def parse_and_validate_url(url: str, settings: Settings) -> tuple[str, str, int]:
    """Validate a user-supplied URL.

    Returns ``(clean_url, hostname, port)``.
    Raises :class:`InvalidUrl`, :class:`BlockedDestination` or :class:`Forbidden`.
    """
    if not isinstance(url, str):
        raise InvalidUrl("La URL debe ser una cadena de texto.")

    candidate = url.strip()
    if not candidate:
        raise InvalidUrl("La URL está vacía.")
    if len(candidate) > MAX_URL_LENGTH:
        raise InvalidUrl(f"La URL supera los {MAX_URL_LENGTH} caracteres.")
    # Reject control characters and embedded whitespace: they enable request
    # smuggling and header injection through some HTTP stacks.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in candidate) or " " in candidate:
        raise InvalidUrl("La URL contiene caracteres de control o espacios.")

    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise InvalidUrl(f"La URL no se pudo interpretar: {exc}") from exc

    if parts.scheme.lower() not in {"http", "https"}:
        raise InvalidUrl(
            "Solo se aceptan URLs http o https.",
            details={"scheme": parts.scheme or None, "accepted": ["http", "https"]},
        )

    hostname = parts.hostname
    if not hostname:
        raise InvalidUrl("La URL no incluye un host válido.")

    if parts.username or parts.password:
        raise InvalidUrl("La URL no puede incluir credenciales (usuario:contraseña@host).")

    # Explicit port handling: urlsplit gives None when the port is the default.
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise InvalidUrl("El puerto de la URL no es válido.") from exc

    if port not in settings.fetch_allowed_ports:
        raise BlockedDestination(
            f"El puerto {port} no está permitido.",
            details={"port": port, "allowed_ports": list(settings.fetch_allowed_ports)},
        )

    host_key = hostname.lower().rstrip(".")
    if host_key in _ALWAYS_BLOCKED_HOSTS:
        raise BlockedDestination(f"El host '{host_key}' está bloqueado por política.")
    if host_key.endswith((".local", ".internal", ".lan", ".intranet")):
        raise BlockedDestination(f"El host '{host_key}' parece un nombre de red interna.")
    if host_key in settings.fetch_blocked_hosts or any(
        host_key.endswith("." + blocked) for blocked in settings.fetch_blocked_hosts
    ):
        raise BlockedDestination(f"El host '{host_key}' está en la lista de bloqueo.")
    if settings.fetch_allowed_hosts and not (
        host_key in settings.fetch_allowed_hosts
        or any(host_key.endswith("." + allowed) for allowed in settings.fetch_allowed_hosts)
    ):
        raise BlockedDestination(
            f"El host '{host_key}' no está en la lista de hosts permitidos.",
            details={"allowed_hosts": list(settings.fetch_allowed_hosts)},
        )

    clean = urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))
    return clean, host_key, port


def _blocked_networks(settings: Settings) -> list[Any]:
    networks = []
    for cidr in settings.fetch_blocked_cidrs:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            # A malformed operator-supplied CIDR must not silently disable protection.
            continue
    return networks


def check_ip(ip_str: str, settings: Settings) -> str | None:
    """Return ``None`` if ``ip_str`` is an acceptable public destination.

    Otherwise return a human-readable reason for the rejection.
    """
    if settings.fetch_allow_private_networks:
        return None

    cleaned = ip_str.split("%", 1)[0]  # strip IPv6 scope id
    try:
        ip = ipaddress.ip_address(cleaned)
    except ValueError:
        return f"la dirección '{ip_str}' no es interpretable"

    # IPv4-mapped IPv6 (::ffff:127.0.0.1) and NAT64 must be judged as IPv4.
    candidates = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            candidates.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            candidates.append(ip.sixtofour)
        if ip.teredo is not None:
            candidates.append(ip.teredo[1])

    for candidate in candidates:
        # Do not rely on `is_global` alone: its meaning shifted across CPython
        # releases (e.g. multicast addresses were considered global before 3.13).
        # Check every special-use property explicitly.
        if (
            candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_private
            or candidate.is_multicast
            or candidate.is_reserved
            or candidate.is_unspecified
            or not candidate.is_global
        ):
            return _reason_for(candidate)
        for network in _blocked_networks(settings):
            if (
                candidate.version == network.version
                and candidate in network
            ):
                return f"la dirección {candidate} pertenece al bloque {network}"
    return None


def _reason_for(ip: Any) -> str:
    if ip.is_loopback:
        return f"la dirección {ip} es de bucle invertido (loopback)"
    if ip.is_link_local:
        return f"la dirección {ip} es de enlace local (incluye el metadata de la nube)"
    if ip.is_private:
        return f"la dirección {ip} pertenece a una red privada"
    if ip.is_multicast:
        return f"la dirección {ip} es multicast"
    if ip.is_reserved or ip.is_unspecified:
        return f"la dirección {ip} está reservada"
    return f"la dirección {ip} no es una dirección pública"


def resolve_host(hostname: str, port: int, settings: Settings) -> tuple[str, ...]:
    """Resolve ``hostname`` and enforce the IP policy on every answer."""
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UpstreamError(
            f"No se pudo resolver el host '{hostname}'.", details={"reason": str(exc)}
        ) from exc

    ips = sorted({info[4][0] for info in infos})
    if not ips:
        raise UpstreamError(f"El host '{hostname}' no devolvió direcciones IP.")

    for ip in ips:
        reason = check_ip(ip, settings)
        if reason is not None:
            raise BlockedDestination(
                "La URL apunta a una dirección de red no permitida.",
                details={"host": hostname, "reason": reason},
            )
    return tuple(ips)


def _peer_ip(response: requests.Response) -> str | None:
    """Best-effort extraction of the socket peer address of a response."""
    raw = getattr(response, "raw", None)
    chains = (
        ("_fp", "fp", "raw", "_sock"),
        ("_connection", "sock"),
        ("_fp", "fp", "_sock"),
    )
    for chain in chains:
        obj: Any = raw
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        getpeername = getattr(obj, "getpeername", None)
        if callable(getpeername):
            try:
                return str(getpeername()[0])
            except (OSError, IndexError):
                return None
    return None


def _read_capped(response: requests.Response, max_bytes: int) -> bytes:
    """Stream the body with a hard cap, aborting as soon as it is exceeded."""
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise PayloadTooLarge(
                    "La imagen remota supera el tamaño máximo permitido.",
                    details={"max_bytes": max_bytes, "declared_bytes": int(declared)},
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        received += len(chunk)
        if received > max_bytes:
            raise PayloadTooLarge(
                "La imagen remota supera el tamaño máximo permitido.",
                details={"max_bytes": max_bytes, "received_bytes": received},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_image_url(
    url: str,
    settings: Settings,
    *,
    session: requests.Session | None = None,
) -> FetchResult:
    """Download an image from ``url`` following the SSRF policy in ``settings``."""
    if not settings.fetch_enabled:
        raise Forbidden(
            "El análisis por URL está deshabilitado en este despliegue.",
            details={"hint": "Establece NID_FETCH_ENABLED=true para habilitarlo."},
        )

    started = time.monotonic()
    deadline = started + settings.fetch_timeout_seconds * (settings.fetch_max_redirects + 1)
    owns_session = session is None
    session = session or requests.Session()
    session.trust_env = False  # never leak corporate proxies/credentials to third parties

    current_url, host, port = parse_and_validate_url(url, settings)
    resolved = resolve_host(host, port, settings)
    hops = [current_url]
    data = b""
    content_type = ""
    peer: str | None = None

    try:
        for _ in range(settings.fetch_max_redirects + 1):
            if time.monotonic() > deadline:
                raise UpstreamTimeout(
                    "Se agotó el tiempo de espera al descargar la imagen.",
                    details={"timeout_seconds": settings.fetch_timeout_seconds},
                )

            remaining = max(0.1, deadline - time.monotonic())
            timeout = (min(settings.fetch_timeout_seconds, remaining),) * 2
            try:
                response = session.get(
                    current_url,
                    headers={
                        "User-Agent": settings.fetch_user_agent,
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    },
                    timeout=timeout,
                    stream=True,
                    allow_redirects=False,
                )
            except requests.exceptions.SSLError as exc:
                raise UpstreamError(
                    "El certificado TLS del servidor remoto no es válido.",
                    details={"reason": type(exc).__name__},
                ) from exc
            except requests.exceptions.Timeout as exc:
                raise UpstreamTimeout(
                    "El servidor remoto tardó demasiado en responder.",
                    details={"timeout_seconds": settings.fetch_timeout_seconds},
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise UpstreamError(
                    "No se pudo conectar con el servidor remoto.",
                    details={"reason": type(exc).__name__},
                ) from exc

            with response:
                peer = _peer_ip(response)
                if peer is not None:
                    reason = check_ip(peer, settings)
                    if reason is not None:
                        # DNS rebinding or a lying resolver: refuse the body.
                        raise BlockedDestination(
                            "La conexión terminó en una dirección de red no permitida.",
                            details={"host": host, "reason": reason},
                        )

                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise UpstreamError(
                            "El servidor devolvió una redirección sin destino.",
                            details={"status_code": response.status_code},
                        )
                    if len(hops) - 1 >= settings.fetch_max_redirects:
                        raise UpstreamError(
                            "Demasiadas redirecciones al obtener la imagen.",
                            details={"max_redirects": settings.fetch_max_redirects},
                        )
                    # Resolve relative redirects against the current hop.
                    next_url = urljoin(current_url, location)
                    current_url, host, port = parse_and_validate_url(next_url, settings)
                    resolved = resolve_host(host, port, settings)
                    hops.append(current_url)
                    continue

                if response.status_code != 200:
                    raise UpstreamError(
                        f"El servidor remoto respondió con el estado {response.status_code}.",
                        details={"status_code": response.status_code},
                    )

                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if content_type in _FORBIDDEN_CONTENT_TYPES:
                    raise UnsupportedMediaType(
                        f"El servidor devolvió '{content_type or 'contenido desconocido'}', no una imagen.",
                        details={"content_type": content_type or None},
                    )

                data = _read_capped(response, settings.fetch_max_bytes)

        if not data:
            raise UpstreamError("El servidor remoto devolvió una respuesta vacía.")

        return FetchResult(
            data=data,
            final_url=current_url,
            status_code=200,
            content_type=content_type or None,
            elapsed_ms=(time.monotonic() - started) * 1000,
            resolved_ips=resolved,
            peer_ip=peer,
            redirects=len(hops) - 1,
            hops=hops,
        )
    finally:
        if owns_session:
            session.close()
