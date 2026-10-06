# syntax=docker/dockerfile:1
#
# Nude Image Detector API
#
# The ONNX model ships inside the `nudenet` wheel, so the image is fully
# self-contained: no downloads at build time beyond pip, and none at runtime.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib2.0-0 is required by the headless OpenCV wheels; curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY wsgi.py gunicorn.conf.py ./

# Never run inference as root.
RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /srv/app
USER apiuser

ENV NID_HOST=0.0.0.0 \
    NID_PORT=8000 \
    NID_ENGINE=nudenet \
    NID_STRICTNESS=balanced \
    NID_LOG_JSON=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

# 2 workers x 4 threads is a sane default for 2 vCPU; tune with NID_GUNICORN_*.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
