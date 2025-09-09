FROM python:3.13-slim AS runtime

WORKDIR /app

COPY requirements.txt requirements.dev.txt* ./
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/* && \
  python -m pip install --upgrade pip wheel setuptools && \
  PIP_NO_CACHE_DIR=0 pip install -r requirements.txt --no-cache-dir --progress-bar off && \
  pip cache purge || true

COPY . /app/

ENV PYTHONPATH=/app/src \
    SERVICE_NAME=transformer \
    SERVICE_TYPE=transformer \
    SERVICE_PORT=8004 \
    TRANSFORMER_SERVICE_PORT=8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

EXPOSE 8004

RUN adduser --disabled-password --gecos "" appuser || useradd -m appuser || true
USER appuser

CMD ["python", "src/main.py", "--mode", "service", "--port", "8004"]
