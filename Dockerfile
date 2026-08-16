FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY ingest.py query.py eval_rag.py ./
COPY sample_docs ./sample_docs

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src"

CMD ["tail", "-f", "/dev/null"]
