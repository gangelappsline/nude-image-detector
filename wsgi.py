"""WSGI entry point.

Used by gunicorn (``gunicorn -c gunicorn.conf.py wsgi:app``), uWSGI or any other
WSGI server.  Kept at the repository root so the usual ``module:app`` convention
works without extra configuration.
"""

from __future__ import annotations

from app import create_app
from app.config import Settings

app = create_app(Settings.from_env())

if __name__ == "__main__":  # pragma: no cover - dev convenience
    app.run(host="0.0.0.0", port=8000, debug=False)
