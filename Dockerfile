# syntax=docker/dockerfile:1.7
# ============================================================
# PaperOdyssey dev image.
#
# The image holds ONLY the environment (JDK for Pyserini/Lucene, CUDA-enabled
# torch, and requirements.txt). The source tree is bind-mounted at run time by
# scripts/../run_docker.sh, so editing code never requires a rebuild.
# Rebuild only when requirements.txt or this file changes.
# ============================================================
# 3.12, not 3.11: Pyserini >= 2.0 declares requires-python >= 3.12. On 3.11 pip
# silently backtracks to an ancient Pyserini. pyproject.toml allows >= 3.11, so
# 3.12 satisfies the project while keeping the sparse retriever current.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# cu128 = NVIDIA wheels (works on Blackwell/RTX 50-series, sm_120) | cpu = CPU-only build
ARG TORCH_VARIANT=cu128
# 1 = install Pyserini (needs the JDK below); 0 = skip it and rely on the
# rank-bm25 fallback declared in retrieval/README.md
ARG INSTALL_PYSERINI=1
# Match the host user so files written into the bind-mounted repo stay editable
ARG UID=1000
ARG GID=1000
ARG USERNAME=dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive

# ---- System packages -------------------------------------------------------
# build-essential + python3-dev: pyjnius (Pyserini's JVM bridge) compiles from source
# libgomp1: OpenMP runtime used by faiss / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        libgomp1 \
        procps \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ---- Temurin JDK 21 (Pyserini / Lucene) ------------------------------------
# Debian bookworm only ships JDK 17; Pyserini's newer releases want 21.
RUN install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
         -o /etc/apt/keyrings/adoptium.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main" \
         > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")" /opt/java

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---- Python: torch first, so the CUDA build wins over the CPU one that
#      sentence-transformers / pyserini would otherwise pull from PyPI --------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"

# ---- Project requirements --------------------------------------------------
# Pyserini is installed separately: it is the one dependency that can fail on a
# fresh machine (JVM bridge compile), and keeping it in its own layer means a
# failure there does not invalidate everything above it.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^pyserini' /tmp/requirements.txt > /tmp/requirements-core.txt \
    && pip install -r /tmp/requirements-core.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_PYSERINI}" = "1" ]; then \
        pip install "$(grep '^pyserini' /tmp/requirements.txt)"; \
    else \
        echo "INSTALL_PYSERINI=0 -> skipping Pyserini (use retrieval.sparse.backend=rankbm25)"; \
    fi

# ---- Non-root user matching the host -------------------------------------
RUN (getent group "${GID}" || groupadd -g "${GID}" "${USERNAME}") \
    && (getent passwd "${UID}" || useradd -u "${UID}" -g "${GID}" -m -s /bin/bash "${USERNAME}") \
    && mkdir -p /workspace && chown "${UID}:${GID}" /workspace

# ---- Runtime environment ---------------------------------------------------
# HF_HOME lives inside the bind-mounted repo (.cache/ is gitignored), so the
# ~4 GB of BGE encoder + reranker weights are downloaded once and survive
# container restarts and image rebuilds.
ENV PYTHONPATH=/workspace \
    HF_HOME=/workspace/.cache/hf \
    TORCH_HOME=/workspace/.cache/torch \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /workspace
USER ${UID}:${GID}

CMD ["bash"]
