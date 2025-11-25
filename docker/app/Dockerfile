# syntax=docker/dockerfile:1.6
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for EasyOCR/OpenCV (CPU). If you have GPU, consider nvidia/cuda base + torch-cuda.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better caching
COPY pyproject.toml .

# Install uv for fast, reproducible installs (optional). Fallback to pip.
RUN pip install --upgrade pip setuptools wheel && \
    pip install "uv==0.4.10" || true

# Install project dependencies
RUN python - <<'PY' > /tmp/requirements.txt
import tomllib
data=tomllib.load(open('pyproject.toml','rb'))
for r in data.get('project',{}).get('dependencies',[]):
    print(r)
PY

RUN if command -v uv >/dev/null 2>&1; then \
      uv pip install --system -r /tmp/requirements.txt; \
    else \
      pip install -r /tmp/requirements.txt; \
    fi

# Download spaCy model (best-effort; ok to fail in offline builds)
RUN python -m spacy download ${SPACY_MODEL:-en_core_web_sm} || true

# Copy source
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ragagent.app:app", "--host", "0.0.0.0", "--port", "8000"]
