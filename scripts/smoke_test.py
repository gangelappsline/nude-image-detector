#!/usr/bin/env python3
"""Smoke test contra una instancia ya arrancada de la API.

Solo usa la biblioteca estándar (ni `requests` ni Pillow), así que puede
ejecutarse dentro del contenedor, en un pipeline de despliegue o desde cualquier
máquina con Python 3.

    python scripts/smoke_test.py --base-url http://127.0.0.1:8000
    python scripts/smoke_test.py --base-url https://api.midominio.com --api-key xxx

Devuelve código de salida 0 si todo pasa y 1 en caso contrario.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import uuid
import zlib

TIMEOUT = 30
RESULTS: list[tuple[bool, str, str]] = []


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def make_png(width: int = 48, height: int = 48, rgb: tuple[int, int, int] = (90, 140, 200)) -> bytes:
    """Construye un PNG válido sin dependencias externas."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def multipart(field: str, filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----nidsmoke{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def call(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict | str, dict]:
    """Devuelve (status, cuerpo_parseado_o_texto, cabeceras)."""
    request = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    request.add_header("User-Agent", "nid-smoke-test/1.0")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read()
            status = response.status
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:  # respuestas 4xx/5xx también se inspeccionan
        raw = exc.read()
        status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}

    try:
        return status, json.loads(raw.decode("utf-8")), response_headers
    except (ValueError, UnicodeDecodeError):
        return status, raw.decode("utf-8", "replace")[:400], response_headers


def check(condition: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((bool(condition), name, detail))
    mark = "✔" if condition else "✘"
    print(f"  {mark} {name}" + (f"  → {detail}" if detail and not condition else ""))
    return bool(condition)


# --------------------------------------------------------------------------- #
# Pruebas
# --------------------------------------------------------------------------- #
def run(base: str, api_key: str | None) -> int:
    base = base.rstrip("/")
    headers = {"X-API-Key": api_key} if api_key else {}
    png = make_png()

    print(f"\nSmoke test contra {base}\n")

    print("Servicio:")
    status, body, _ = call(f"{base}/health", headers=headers)
    check(status == 200 and isinstance(body, dict) and body.get("engine"), "GET /health")

    status, body, _ = call(f"{base}/ready", headers=headers)
    check(status == 200 and isinstance(body, dict) and body.get("ready") is True, "GET /ready")
    if isinstance(body, dict):
        print(f"      motor={body.get('engine')} modelo={body.get('model')} versión={body.get('version')}")

    status, body, _ = call(f"{base}/v1/info", headers=headers)
    check(status == 200 and isinstance(body, dict) and "profiles" in body, "GET /v1/info")

    status, body, _ = call(f"{base}/v1/labels", headers=headers)
    check(
        status == 200 and isinstance(body, dict) and body.get("label_count", 0) >= 17,
        "GET /v1/labels",
    )

    status, body, _ = call(f"{base}/openapi.json", headers=headers)
    check(
        status == 200 and isinstance(body, dict) and "/v1/analyze" in body.get("paths", {}),
        "GET /openapi.json",
    )

    status, body, _ = call(f"{base}/", headers=headers)
    check(status == 200 and "Nude Image Detector" in str(body), "GET / (documentación)")

    print("\nAnálisis:")
    body_bytes, content_type = multipart("file", "humo.png", png, "image/png")
    status, payload, response_headers = call(
        f"{base}/v1/analyze",
        method="POST",
        data=body_bytes,
        headers={**headers, "Content-Type": content_type},
    )
    ok = check(status == 200 and isinstance(payload, dict) and payload.get("verdict") in {"allow", "review", "block"},
               "POST /v1/analyze (multipart)")
    if ok and isinstance(payload, dict):
        print(
            f"      verdict={payload.get('verdict')} risk={payload.get('risk_score')} "
            f"ms={payload.get('elapsed_ms')} request_id={payload.get('request_id')}"
        )
        check(
            response_headers.get("X-Request-ID") == payload.get("request_id"),
            "Cabecera X-Request-ID coincide con el cuerpo",
        )

    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=json.dumps({"image_base64": base64.b64encode(png).decode()}).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    check(status == 200 and isinstance(payload, dict) and payload.get("source", {}).get("type") == "base64",
          "POST /v1/analyze (base64)")

    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=png,
        headers={**headers, "Content-Type": "image/png"},
    )
    check(status == 200 and isinstance(payload, dict), "POST /v1/analyze (cuerpo binario)")

    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=json.dumps({}).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    check(
        status == 400 and isinstance(payload, dict) and payload.get("error", {}).get("code") == "missing_image",
        "POST /v1/analyze sin imagen → 400 missing_image",
    )

    body_bytes, content_type = multipart("file", "no-es-imagen.png", b"contenido no binario de imagen", "image/png")
    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=body_bytes,
        headers={**headers, "Content-Type": content_type},
    )
    check(
        status == 422 and isinstance(payload, dict) and payload.get("error", {}).get("code") == "unprocessable_image",
        "Archivo que no es imagen → 422 unprocessable_image",
    )

    print("\nLotes:")
    raw = base64.b64encode(png).decode()
    status, payload, _ = call(
        f"{base}/v1/analyze/batch",
        method="POST",
        data=json.dumps({"items": [{"image_base64": raw}, {"image_base64": raw}]}).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    check(
        status == 200 and isinstance(payload, dict) and payload.get("summary", {}).get("succeeded") == 2,
        "POST /v1/analyze/batch (2 elementos)",
    )

    print("\nSeguridad:")
    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=json.dumps({"url": "http://169.254.169.254/latest/meta-data/"}).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    code = payload.get("error", {}).get("code") if isinstance(payload, dict) else None
    check(
        status == 403 and code in {"blocked_destination", "forbidden"},
        "URL al metadata de la nube → 403 (anti-SSRF)",
        f"status={status} code={code}",
    )

    status, payload, _ = call(
        f"{base}/v1/analyze",
        method="POST",
        data=json.dumps({"url": "http://127.0.0.1:8000/health"}).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    code = payload.get("error", {}).get("code") if isinstance(payload, dict) else None
    check(
        status == 403 and code in {"blocked_destination", "forbidden"},
        "URL a loopback → 403 (anti-SSRF)",
        f"status={status} code={code}",
    )

    status, _, response_headers = call(f"{base}/health")
    check(
        response_headers.get("X-Content-Type-Options") == "nosniff"
        and "Content-Security-Policy" in response_headers,
        "Cabeceras de seguridad presentes",
    )

    failures = [name for passed, name, _ in RESULTS if not passed]
    total = len(RESULTS)
    print(f"\n{'─' * 62}")
    print(f"  {total - len(failures)}/{total} comprobaciones correctas")
    if failures:
        print("  Fallidas:")
        for name in failures:
            print(f"    - {name}")
    print(f"{'─' * 62}\n")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test de la API nude-image-detector")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="URL base del servicio")
    parser.add_argument("--api-key", default=None, help="Valor para la cabecera X-API-Key")
    args = parser.parse_args()
    try:
        return run(args.base_url, args.api_key)
    except urllib.error.URLError as exc:
        print(f"\n✘ No se pudo conectar con {args.base_url}: {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":
    sys.exit(main())
