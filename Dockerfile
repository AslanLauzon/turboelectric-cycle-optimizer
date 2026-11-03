# Hugging Face Space: serves the marimo demonstrator on port 7860.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY --chown=user . .

EXPOSE 7860
CMD ["marimo", "run", "ASTROMECHANICA OPTIMIZER.py", "--host", "0.0.0.0", "--port", "7860", "--headless", "--no-sandbox"]
