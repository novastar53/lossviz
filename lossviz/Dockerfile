FROM python:3.7.2-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir lossviz
COPY src lossviz/.
COPY pyproject.toml .

RUN uv add torch jax optax losscape

