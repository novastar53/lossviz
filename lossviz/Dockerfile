FROM python:slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml .
RUN uv add torch torchvision jax flax optax losscape 

RUN mkdir lossviz
WORKDIR /lossviz

COPY src/lossviz .

ENTRYPOINT ["uv", "run", "main.py"]