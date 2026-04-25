FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Pin JAX with CUDA 12 wheels. Pin econirl from a known commit so the
# image is reproducible.
ARG ECONIRL_REF=main
WORKDIR /workspace

# Install econirl from the GitHub repository at the pinned ref.
RUN pip install --upgrade pip && \
    pip install "jax[cuda12_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install "git+https://github.com/rawatpranjal/econirl.git@${ECONIRL_REF}#egg=econirl"

# Drop in the experiment package so that `python -m
# experiments.jss_deep_run.run_cell` resolves. The Dockerfile lives
# inside the experiments tree so the build context already contains
# this directory.
COPY .. /workspace/experiments_src
RUN pip install -e /workspace/experiments_src

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "experiments.jss_deep_run.run_cell"]
