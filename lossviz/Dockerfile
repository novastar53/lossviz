FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get clean && apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libmpich-dev libopenmpi-dev  #python3-pip

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

RUN mkdir lossviz
COPY src lossviz/.

WORKDIR lossviz

RUN uv init 
RUN uv add torch
RUN uv add mpi4py
RUN uv add numpy
RUN uv add h5py
RUN uv add matplotlib
RUN uv add scipy


