FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Install dependencies before copying the code so edits don't invalidate this layer.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user data/samples ./data/samples
RUN uv sync --locked --no-dev

# Bake the embedding model into the image so the first question is not slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/healthz')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
