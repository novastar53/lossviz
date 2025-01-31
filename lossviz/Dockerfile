FROM python:slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir lossviz
COPY src/lossviz/* lossviz/
COPY pyproject.toml .

RUN uv add torch jax optax losscape

#RUN pwd

ENTRYPOINT ["uv", "run", "--", "python", "lossviz/main.py"]