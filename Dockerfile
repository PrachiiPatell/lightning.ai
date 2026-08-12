# SPT-DALES inference server.
#
# Every version pin below is load-bearing. They were arrived at by debugging a
# from-scratch install on Ubuntu 24.04 + L4, and the failures were not obvious:
#
#   * CUDA 12.1 (what torch 2.2.0 is built against) is NOT available from
#     NVIDIA's apt repo for Ubuntu 24.04 "noble" -- 12.5 is the oldest. Using
#     the devel image below sidesteps the whole apt-repo problem by shipping
#     nvcc in the base layer.
#   * GCC 13 (Ubuntu 24.04's default) CANNOT compile torch 2.2.0's headers:
#     ATen/core/boxing/impl/boxing.h fails with "expected primary-expression
#     before '>' token". GCC 12 is required, and nvcc must be told to use it
#     via NVCC_PREPEND_FLAGS -- setting CC/CXX alone does not reach nvcc's
#     internal host-compiler choice.
#   * The pip `nvidia-cuda-nvcc-cu12` wheel ships only ptxas, not nvcc, so the
#     no-root pip fallback cannot build FRNN at all.
#   * numpy must stay <2: torch 2.2.0 and open3d link the 1.x ABI.
#
# Build (from this directory):
#   docker build -t spt-dales:1.0 .
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v /home/ubuntu/spt_dales.ckpt:/root/spt_dales.ckpt:ro \
#     --restart unless-stopped --name spt spt-dales:1.0

# devel (not runtime) — nvcc is needed to compile FRNN.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9

# 8.9 = L4 / RTX 4090 (Ada). Change to match the deployment GPU:
#   7.5 = T4    8.0 = A100    8.6 = A10G    9.0 = H100
# Building for the wrong arch produces kernels that will not run.

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git build-essential ninja-build \
        gcc-12 g++-12 \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# nvcc must use GCC 12 (see header note above).
ENV CC=/usr/bin/gcc-12 \
    CXX=/usr/bin/g++-12 \
    NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ── torch first: everything below compiles or links against it ──────────────
RUN pip install --no-cache-dir \
        torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121

# numpy pinned BEFORE the rest so no transitive dep pulls in 2.x.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" "scipy==1.14.1" "laspy[lazrs]" tqdm pyyaml

RUN pip install --no-cache-dir \
        torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN pip install --no-cache-dir \
        pytorch-lightning "torchmetrics==0.11.4" \
        hydra-core hydra-colorlog hydra-submitit-launcher omegaconf \
        "torch_geometric==2.3.0" \
        h5py numba "plyfile<1.1" colorhash \
        "plotly==5.9.0" matplotlib seaborn \
        "rich<=14.0" wandb open3d GitPython

# src.utils imports these at module load even on a pure inference path.
RUN pip install --no-cache-dir \
        ipyfilechooser "ipywidgets>=7.6" ipykernel \
        "jupyterlab>=3" "notebook>=5.3" jupyter-dash torch_tb_profiler

RUN pip install --no-cache-dir \
        pycut-pursuit pygrid-graph pgeof \
        torch-graph-components torch-ransac3d pyrootutils gdown

RUN pip install --no-cache-dir \
        pyg_lib torch_cluster \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

# ── superpoint_transformer + FRNN ───────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/drprojects/superpoint_transformer.git

# FRNN's path is NOT free-form: src/utils/neighbors.py does
#   from src.dependencies.FRNN import frnn
# at import time, so it must live exactly here.
WORKDIR /opt/superpoint_transformer
RUN git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN

# --no-build-isolation is required: FRNN's setup.py does `import torch` at build
# time, and pip's isolated build env has no torch. setuptools<70 because newer
# versions break these older setup.py files.
RUN pip install --no-cache-dir "setuptools<70" wheel ninja \
    && cd src/dependencies/FRNN/external/prefix_sum \
    && pip install --no-build-isolation --no-cache-dir . \
    && cd ../.. \
    && pip install --no-build-isolation --no-cache-dir . \
    && cd /opt \
    && python -c "from frnn import _C; print('FRNN OK')"

# Installs above can quietly re-upgrade numpy; put it back and verify.
RUN pip install --no-cache-dir "numpy==1.26.4" "plyfile<1.1" \
    && python -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__"

COPY spt_server.py /opt/spt_server.py

# The checkpoint is NOT baked in — mount it at run time so the image stays
# reusable across model versions:
#   -v /path/to/spt_dales.ckpt:/root/spt_dales.ckpt:ro
ENV SPT_ROOT=/opt/superpoint_transformer

WORKDIR /opt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "spt_server:app", "--host", "0.0.0.0", "--port", "8000"]
