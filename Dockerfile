FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/home/rigorousrag

WORKDIR /app

RUN groupadd --system rigorousrag \
    && useradd --system --gid rigorousrag --create-home --home-dir /home/rigorousrag rigorousrag \
    && apt-get update \
    && apt-get install --yes --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY --chown=rigorousrag:rigorousrag . /app
RUN mkdir -p /app/uploads /app/rag_storage /app/data /home/rigorousrag/.cache \
    && chown -R rigorousrag:rigorousrag /app/uploads /app/rag_storage /app/data /home/rigorousrag

USER rigorousrag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
  CMD ["python", "-m", "tools.healthcheck"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
