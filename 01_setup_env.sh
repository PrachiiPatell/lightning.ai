#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lightning.ai Studio — LiDAR deep learning environment
#
# Mirrors the versions already proven in Dockerfile_lidar (PyTorch 2.1.0 /
# CUDA 12.1 / spconv-cu121 / numpy 1.26.4), because OpenPCDet, spconv and
# torch-scatter must all be built against the SAME torch+CUDA pair. Installing
# "latest" for any one of them is the usual cause of an import-time crash that
# only shows up after a long build.
#
# Run once per Studio:   bash 01_setup_env.sh
# Safe to re-run: every step is idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# ── 0. Sanity: are we on a GPU machine? ────────────────────────────────────
log "Checking GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found. Switch this Studio to a GPU machine before running."
  warn "Continuing anyway — CPU-only installs will succeed but training will not."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Detect the CUDA arch of the attached GPU so spconv/OpenPCDet compile kernels
# for it. Hardcoding 7.5 (as the Dockerfile does) silently produces a build
# that will not run on A100 (8.0) or L4/A10G (8.6).
if command -v nvidia-smi >/dev/null 2>&1; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  export TORCH_CUDA_ARCH_LIST="${CC}"
  log "Detected compute capability ${CC} → TORCH_CUDA_ARCH_LIST=${CC}"
else
  export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
  warn "No GPU detected; building for 7.5;8.0;8.6"
fi

# ── 1. System packages ─────────────────────────────────────────────────────
log "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git build-essential ninja-build \
  libgl1 libglib2.0-0 \
  >/dev/null
echo "ok"

# ── 2. PyTorch (CUDA 12.1 build) ───────────────────────────────────────────
# Your Dockerfile uses 2.1.0, but the cu121 wheel index no longer publishes it
# (earliest is now 2.2.0). 2.2.0 is the closest still-available version and is
# ABI-compatible with the rest of this stack. Override with TORCH_VERSION=... .
TORCH_VERSION="${TORCH_VERSION:-2.2.0}"
case "$TORCH_VERSION" in
  2.2.0) TV=0.17.0; TA=2.2.0 ;;
  2.2.1) TV=0.17.1; TA=2.2.1 ;;
  2.2.2) TV=0.17.2; TA=2.2.2 ;;
  2.3.0) TV=0.18.0; TA=2.3.0 ;;
  2.3.1) TV=0.18.1; TA=2.3.1 ;;
  2.4.0) TV=0.19.0; TA=2.4.0 ;;
  2.4.1) TV=0.19.1; TA=2.4.1 ;;
  2.5.0) TV=0.20.0; TA=2.5.0 ;;
  2.5.1) TV=0.20.1; TA=2.5.1 ;;
  *) warn "Unknown torch $TORCH_VERSION — letting pip resolve torchvision/torchaudio"
     TV=""; TA="" ;;
esac

log "PyTorch $TORCH_VERSION + CUDA 12.1"
if [ -n "$TV" ]; then
  pip install --quiet \
    "torch==${TORCH_VERSION}" "torchvision==${TV}" "torchaudio==${TA}" \
    --index-url https://download.pytorch.org/whl/cu121
else
  pip install --quiet "torch==${TORCH_VERSION}" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print(f'  torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')"

# ── 3. Core scientific stack ───────────────────────────────────────────────
# numpy is pinned BELOW 2.0 on purpose: OpenPCDet, spconv and open3d all still
# link against the 1.x ABI, and numpy 2 breaks them at import.
#
# scipy matters more than it looks: the SPT server projects voxel predictions
# back to every original point with scipy.spatial.cKDTree.
log "Scientific stack (numpy pinned <2.0)"
pip install --quiet \
  "numpy==1.26.4" \
  "scipy==1.14.1" \
  "laspy[lazrs]" \
  tqdm pyyaml
echo "ok"

# ── 4. OpenPCDet-only extras ───────────────────────────────────────────────
# Skipped by default: the SPT inference server does not import any of these.
# Set WITH_OPENPCDET=1 if you also want the PointPillars/CenterPoint path.
if [ "${WITH_OPENPCDET:-0}" = "1" ]; then
  log "OpenPCDet extras (spconv, open3d, kornia, av2)"
  pip install --quiet \
    scikit-image SharedArray easydict tensorboard \
    open3d opencv-python-headless \
    "kornia==0.6.12" av2
  pip install --quiet spconv-cu121
  python -c "import spconv; print('  spconv', spconv.__version__)"
else
  log "Skipping OpenPCDet extras (set WITH_OPENPCDET=1 to include them)"
fi

# ── 5. torch-scatter / torch-sparse ────────────────────────────────────────
# Needed by SPT and most graph-based point cloud models. These MUST come from
# the wheel index matching the exact torch build, or pip falls back to a source
# build that takes ~20 minutes and often fails.
# Derive the wheel index from the torch that actually got installed — a
# hardcoded URL silently falls back to a ~20 min source build (often failing)
# the moment the torch pin moves.
PYG_TAG="$(python -c 'import torch,re; v=torch.__version__.split("+")[0]; print(f"torch-{v}+cu121")')"
log "torch-scatter / torch-sparse (wheels for ${PYG_TAG})"
# These ship compiled C++ extensions linked against a SPECIFIC torch build. If
# torch changes underneath them, pip sees "already installed" and skips, leaving
# binaries whose symbols no longer resolve:
#   undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs
# Uninstall first so the correct wheels are always fetched for the current torch.
pip uninstall -y -q torch-scatter torch-sparse torch-cluster pyg-lib 2>/dev/null || true
if ! pip install --quiet --no-cache-dir torch-scatter torch-sparse \
      -f "https://data.pyg.org/whl/${PYG_TAG}.html"; then
  warn "No prebuilt wheels for ${PYG_TAG}."
  warn "Check https://data.pyg.org/whl/ for a published tag, then re-run with"
  warn "  TORCH_VERSION=<that version> bash 01_setup_env.sh"
  warn "Continuing — SPT needs these, OpenPCDet does not."
fi
python -c "import torch_scatter, torch_sparse; print('  torch-scatter/sparse ok')" 2>/dev/null \
  || warn "torch-scatter/torch-sparse not importable"

log "Base environment ready"
echo "Next: bash 02_install_openpcdet.sh"
