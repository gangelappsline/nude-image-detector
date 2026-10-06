"""Development entry point.

``python run.py`` starts Flask's built-in server with the threaded worker, which
is fine for local experiments.  For real traffic use gunicorn::

    gunicorn -c gunicorn.conf.py wsgi:app
"""

from __future__ import annotations

from app import create_app
from app.config import Settings


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    print(
        f"\n  Nude Image Detector {settings.version}\n"
        f"  motor: {settings.engine} · perfil: {settings.strictness}\n"
        f"  escuchando en http://{settings.host}:{settings.port}\n"
        f"  docs en http://127.0.0.1:{settings.port}/\n"
    )
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        threaded=True,
        use_reloader=settings.debug,
    )


if __name__ == "__main__":
    main()
