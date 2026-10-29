# Override to use a mirror, e.g. public.ecr.aws/docker/library/python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface

# Dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY eval ./eval
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "manual_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
