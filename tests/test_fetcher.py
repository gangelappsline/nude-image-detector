"""SSRF protection and remote fetching.

These tests never touch the internet: hostname validation is checked directly,
DNS is stubbed where needed, and the happy paths run against a loopback server.
"""

from __future__ import annotations

import socket

import pytest

from app.config import Settings
from app.core.fetcher import check_ip, fetch_image_url, parse_and_validate_url, resolve_host
from app.errors import (
    BlockedDestination,
    Forbidden,
    InvalidUrl,
    PayloadTooLarge,
    UnsupportedMediaType,
    UpstreamError,
    UpstreamTimeout,
)

from .conftest import base_settings


@pytest.fixture
def settings() -> Settings:
    return base_settings(fetch_enabled=True, fetch_allow_private_networks=False)


@pytest.fixture
def open_settings(local_server: tuple[str, int]) -> Settings:
    """Settings that permit loopback so the test server is reachable."""
    _, port = local_server
    return base_settings(
        fetch_enabled=True,
        fetch_allow_private_networks=True,
        fetch_allowed_ports=(port, 80, 443),
        fetch_timeout_seconds=3.0,
        fetch_max_bytes=5 * 1024 * 1024,
    )


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/a.jpg",
        "file:///etc/passwd",
        "gopher://example.com/",
        "dict://example.com/",
        "javascript:alert(1)",
        "data:image/png;base64,AAAA",
        "example.com/a.jpg",
        "http:///no-host.jpg",
        "http://",
    ],
)
def test_rejects_non_http_schemes_and_malformed_urls(settings: Settings, url: str) -> None:
    with pytest.raises(InvalidUrl):
        parse_and_validate_url(url, settings)


def test_rejects_empty_and_oversized_urls(settings: Settings) -> None:
    with pytest.raises(InvalidUrl):
        parse_and_validate_url("   ", settings)
    with pytest.raises(InvalidUrl):
        parse_and_validate_url("http://example.com/" + "a" * 3000, settings)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/\r\nX-Injected: 1",
        "http://example.com/a b.jpg",
        "http://example.com/\x00.jpg",
    ],
)
def test_rejects_control_characters_and_spaces(settings: Settings, url: str) -> None:
    with pytest.raises(InvalidUrl):
        parse_and_validate_url(url, settings)


def test_rejects_embedded_credentials(settings: Settings) -> None:
    with pytest.raises(InvalidUrl, match="credenciales"):
        parse_and_validate_url("http://user:pass@example.com/a.jpg", settings)


def test_accepts_plain_http_and_https(settings: Settings) -> None:
    clean, host, port = parse_and_validate_url("http://example.com/a.jpg", settings)
    assert (clean, host, port) == ("http://example.com/a.jpg", "example.com", 80)

    clean, host, port = parse_and_validate_url("HTTPS://Example.COM/a.jpg", settings)
    assert host == "example.com" and port == 443


def test_strips_fragment(settings: Settings) -> None:
    clean, _, _ = parse_and_validate_url("http://example.com/a.jpg#section", settings)
    assert "#" not in clean


def test_rejects_unlisted_ports(settings: Settings) -> None:
    with pytest.raises(BlockedDestination, match="puerto"):
        parse_and_validate_url("http://example.com:8080/a.jpg", settings)


def test_accepts_listed_ports() -> None:
    settings = base_settings(fetch_allowed_ports=(80, 443, 8080))
    _, _, port = parse_and_validate_url("http://example.com:8080/a.jpg", settings)
    assert port == 8080


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST",
        "ip6-localhost",
        "metadata",
        "metadata.google.internal",
        "printer.local",
        "nas.lan",
        "intranet.internal",
    ],
)
def test_blocks_internal_hostnames(settings: Settings, host: str) -> None:
    with pytest.raises(BlockedDestination):
        parse_and_validate_url(f"http://{host}/a.jpg", settings)


def test_blocked_hosts_setting(settings: Settings) -> None:
    from dataclasses import replace

    strict = replace(settings, fetch_blocked_hosts=("evil.test", "cdn.bad.test"))
    with pytest.raises(BlockedDestination):
        parse_and_validate_url("http://evil.test/a.jpg", strict)
    with pytest.raises(BlockedDestination):
        parse_and_validate_url("http://sub.cdn.bad.test/a.jpg", strict)
    parse_and_validate_url("http://good.test/a.jpg", strict)


def test_allowed_hosts_setting_acts_as_allowlist() -> None:
    settings = base_settings(fetch_allowed_hosts=("cdn.example.com",))
    parse_and_validate_url("http://cdn.example.com/a.jpg", settings)
    parse_and_validate_url("http://img.cdn.example.com/a.jpg", settings)
    with pytest.raises(BlockedDestination):
        parse_and_validate_url("http://other.example.com/a.jpg", settings)


# --------------------------------------------------------------------------- #
# IP policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "127.1.2.3",
        "::1",
        "0.0.0.0",
        "::",
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.0.1",
        "169.254.169.254",  # cloud metadata
        "169.254.1.1",
        "100.64.0.1",  # CGNAT
        "198.18.0.1",  # benchmarking
        "192.0.0.1",  # IETF
        "224.0.0.1",  # multicast
        "240.0.0.1",  # reserved
        "255.255.255.255",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:169.254.169.254",
        "::ffff:10.0.0.1",
        "fc00::1",  # IPv6 unique local
        "fe80::1",  # IPv6 link local
        "fe80::1%eth0",  # with scope id
    ],
)
def test_blocks_non_public_addresses(settings: Settings, ip: str) -> None:
    assert check_ip(ip, settings) is not None, f"{ip} debería estar bloqueada"


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_allows_public_addresses(settings: Settings, ip: str) -> None:
    assert check_ip(ip, settings) is None


def test_allow_private_networks_disables_the_guard() -> None:
    permissive = base_settings(fetch_allow_private_networks=True)
    assert check_ip("127.0.0.1", permissive) is None
    assert check_ip("169.254.169.254", permissive) is None


def test_blocked_cidrs_are_honoured() -> None:
    settings = base_settings(fetch_blocked_cidrs=("93.184.216.0/24",))
    assert check_ip("93.184.216.34", settings) is not None
    assert check_ip("8.8.8.8", settings) is None


def test_malformed_blocked_cidr_does_not_disable_protection() -> None:
    settings = base_settings(fetch_blocked_cidrs=("not-a-cidr",))
    assert check_ip("127.0.0.1", settings) is not None


# --------------------------------------------------------------------------- #
# DNS resolution
# --------------------------------------------------------------------------- #
def test_resolve_host_rejects_private_answers(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """A public hostname pointing at loopback (DNS rebinding) must be refused."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(BlockedDestination, match="no permitida"):
        resolve_host("looks-public.test", 80, settings)


def test_resolve_host_accepts_public_answers(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        ],
    )
    assert resolve_host("example.test", 80, settings) == ("93.184.216.34",)


def test_resolve_host_surfaces_dns_failures(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(UpstreamError, match="resolver"):
        resolve_host("does-not-exist.test", 80, settings)


def test_resolve_host_rejects_empty_answers(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(UpstreamError):
        resolve_host("empty.test", 80, settings)


# --------------------------------------------------------------------------- #
# Real fetching (against the loopback test server)
# --------------------------------------------------------------------------- #
def test_fetch_disabled_is_refused(local_base_url: str) -> None:
    settings = base_settings(fetch_enabled=False)
    with pytest.raises(Forbidden, match="deshabilitado"):
        fetch_image_url(f"{local_base_url}/image", settings)


def test_fetch_image_happy_path(open_settings: Settings, local_base_url: str) -> None:
    result = fetch_image_url(f"{local_base_url}/image", open_settings)
    assert result.status_code == 200
    assert result.content_type == "image/jpeg"
    assert result.data[:3] == b"\xff\xd8\xff"
    assert result.redirects == 0
    assert result.elapsed_ms >= 0


def test_fetch_follows_redirects(open_settings: Settings, local_base_url: str) -> None:
    result = fetch_image_url(f"{local_base_url}/hop", open_settings)
    assert result.data[:3] == b"\xff\xd8\xff"
    assert result.redirects == 1
    assert result.final_url.endswith("/image")


def test_fetch_follows_relative_redirects(open_settings: Settings, local_base_url: str) -> None:
    result = fetch_image_url(f"{local_base_url}/relative", open_settings)
    assert result.data[:3] == b"\xff\xd8\xff"


def test_redirect_loop_is_stopped(open_settings: Settings, local_base_url: str) -> None:
    with pytest.raises(UpstreamError, match="redirecciones"):
        fetch_image_url(f"{local_base_url}/loop", open_settings)


def test_html_response_is_refused(open_settings: Settings, local_base_url: str) -> None:
    with pytest.raises(UnsupportedMediaType):
        fetch_image_url(f"{local_base_url}/html", open_settings)


def test_oversized_response_is_refused(local_server: tuple[str, int]) -> None:
    _, port = local_server
    settings = base_settings(
        fetch_allow_private_networks=True,
        fetch_allowed_ports=(port,),
        fetch_max_bytes=1024,
    )
    with pytest.raises(PayloadTooLarge):
        fetch_image_url(f"http://127.0.0.1:{port}/huge", settings)


def test_streaming_cap_without_content_length(local_server: tuple[str, int]) -> None:
    """Even a lying/absent Content-Length must not let the body grow forever."""
    _, port = local_server
    settings = base_settings(
        fetch_allow_private_networks=True,
        fetch_allowed_ports=(port,),
        fetch_max_bytes=16,
    )
    with pytest.raises(PayloadTooLarge):
        fetch_image_url(f"http://127.0.0.1:{port}/image", settings)


def test_error_status_is_reported(open_settings: Settings, local_base_url: str) -> None:
    with pytest.raises(UpstreamError, match="404"):
        fetch_image_url(f"{local_base_url}/missing", open_settings)


def test_empty_body_is_reported(open_settings: Settings, local_base_url: str) -> None:
    with pytest.raises(UpstreamError, match="vacía"):
        fetch_image_url(f"{local_base_url}/empty", open_settings)


def test_timeout_is_reported(local_server: tuple[str, int]) -> None:
    _, port = local_server
    settings = base_settings(
        fetch_allow_private_networks=True,
        fetch_allowed_ports=(port,),
        fetch_timeout_seconds=0.5,
    )
    with pytest.raises(UpstreamTimeout):
        fetch_image_url(f"http://127.0.0.1:{port}/slow", settings)


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1:{port}/image",
        "http://localhost:{port}/image",
        "http://[::1]:{port}/image",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/image",
        "http://192.168.1.1/router.png",
    ],
)
def test_private_targets_are_refused_even_when_the_port_is_allowed(
    local_server: tuple[str, int], target: str
) -> None:
    _, port = local_server
    settings = base_settings(
        fetch_enabled=True,
        fetch_allow_private_networks=False,
        fetch_allowed_ports=(port, 80, 443),
    )
    with pytest.raises(BlockedDestination):
        fetch_image_url(target.format(port=port), settings)


def test_describe_never_includes_query_secrets(open_settings: Settings, local_base_url: str) -> None:
    result = fetch_image_url(f"{local_base_url}/image", open_settings)
    result.final_url = f"{local_base_url}/image?token=super-secret"
    described = result.describe()
    assert described["host"] == "127.0.0.1"
    assert "super-secret" not in described["host"]
