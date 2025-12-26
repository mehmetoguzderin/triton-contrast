# CUDA + cuDNN base for devcontainer workstations
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal OS deps for building/installing Python packages + torchvision image codecs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      bash \
      build-essential \
      imagemagick \
      pkg-config \
      python3.12 \
      python3.12-venv \
      python3-pip \
      python3.12-dev \
      libjpeg-turbo8-dev \
      libpng-dev \
      sudo \
      tini \
    && rm -rf /var/lib/apt/lists/*

USER root
ENV PATH="/root/.local/bin:${PATH}"

# Install uv (official installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Reduce zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "sleep infinity"]
