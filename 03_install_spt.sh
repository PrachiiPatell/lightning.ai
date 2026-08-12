#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Superpoint Transformer (SPT) + DALES — semantic segmentation for aerial LiDAR.
#
# This is the stack your inference server actually uses. Its imports are:
#   torch, laspy, numpy, scipy (cKDTree), fastapi, pydantic, uvicorn,
#   hydra, omegaconf, and SPT's own src.{models,data,transforms}.
#
# Note it does NOT use spconv, open3d, pcdet, kornia or av2 — those belong to
# the OpenPCDet/PointPillars path in 02_install_openpcdet.sh. Install this
# script alone if SPT is all you need.
#
# Run AFTER 01_setup_env.sh:   bash 03_install_spt.sh
# ---------------------------------------------------------------------------
set -euo pipefail

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

# The server hardcodes this path in sys.path.insert and os.chdir, so the clone
# location is not free-form — it must land exactly here.
WORKSPACE="/teamspace/studios/this_studio"
if [ ! -d "$WORKSPACE" ]; then
  warn "$WORKSPACE not found (not a Lightning Studio?). Falling back to \$HOME."
  warn "If so, update the sys.path.insert/os.chdir paths at the top of the server."
  WORKSPACE="$HOME"
fi
cd "$WORKSPACE"
log "Workspace: $WORKSPACE"

if command -v nvidia-smi >/dev/null 2>&1; then
  export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
else
  export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ── 1. Clone ───────────────────────────────────────────────────────────────
if [ ! -d superpoint_transformer ]; then
  log "Cloning superpoint_transformer"
  git clone https://github.com/drprojects/superpoint_transformer.git
fi
cd superpoint_transformer

# ── 2. Python dependencies ─────────────────────────────────────────────────
# hydra/omegaconf drive the config compose() the server calls; pytorch-lightning
# provides the LightningModule base that SemanticSegmentationModule extends.
# Versions mirror the repo's install.sh where it pins them (torchmetrics
# 0.11.4, torch_geometric 2.3.0, rich<=14.0) -- SPT's LightningModule and
# config schema are written against those, and newer majors change APIs it
# calls.
log "Python dependencies"
pip install --quiet \
  pytorch-lightning "torchmetrics==0.11.4" \
  "hydra-core" hydra-colorlog hydra-submitit-launcher omegaconf \
  "torch_geometric==2.3.0" \
  h5py numba "plyfile<1.1" colorhash \
  "plotly==5.9.0" matplotlib seaborn \
  "rich<=14.0" tqdm pyyaml wandb \
  open3d GitPython \
  || warn "some deps failed — re-run or install individually"

# SPT's shared modules import notebook/visualisation packages at module load
# even on a pure inference path (e.g. ipyfilechooser via src.utils), so these
# are required despite never being used by the server itself.
log "Notebook/visualisation deps (imported unconditionally by src.utils)"
pip install --quiet \
  ipyfilechooser "ipywidgets>=7.6" ipykernel \
  "jupyterlab>=3" "notebook>=5.3" jupyter-dash \
  torch_tb_profiler \
  || warn "some notebook deps failed"

# The inference server itself needs these; installing here so one script gets
# you from empty Studio to a serving endpoint.
log "Serving stack (fastapi/uvicorn/pydantic)"
pip install --quiet fastapi "uvicorn[standard]" pydantic

# ── 3. FRNN — fixed-radius neighbour search, compiled against this GPU ──────
#
# FRNN compiles real .cu files, so torch's cpp_extension enforces that the
# SYSTEM nvcc matches the CUDA torch was built with (12.1). Lightning Studios
# ship CUDA 13.0 as the default toolkit, which trips that hard check.
#
# Look for a 12.x toolkit and point the build at it. If none exists, skip the
# build entirely: FRNN is a SPEED optimisation, and SPT falls back to a slower
# neighbour search without it. A failed FRNN is not a blocker for inference.
log "Locating a CUDA toolkit matching torch"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '')"
TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
CUDA_MATCH=""
for d in /usr/local/cuda-${TORCH_CUDA} /usr/local/cuda-${TORCH_CUDA_MAJOR}.* /usr/local/cuda; do
  [ -x "$d/bin/nvcc" ] || continue
  _v="$("$d/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  if [ "${_v%%.*}" = "$TORCH_CUDA_MAJOR" ]; then
    CUDA_MATCH="$d"; log "  found CUDA $_v at $d (torch wants $TORCH_CUDA)"; break
  fi
done

SKIP_FRNN=0
if [ -z "$CUDA_MATCH" ]; then
  _sys="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  warn "No CUDA ${TORCH_CUDA_MAJOR}.x toolkit found (system nvcc reports ${_sys:-none})."
  warn "FRNN is NOT optional: src/utils/neighbors.py does"
  warn "  'from src.dependencies.FRNN import frnn'"
  warn "at import time, so SPT will not load without it."

  # Install the matching toolkit via pip. nvidia-cuda-nvcc-cuXX ships just the
  # compiler (~100 MB) rather than the full apt toolkit (~3 GB), and needs no
  # root. That is enough for torch's cpp_extension to build FRNN.
  # nvcc alone is not enough: torch's cpp_extension also needs the CUDA runtime
  # HEADERS (cuda_runtime.h et al), which ship in a separate wheel.
  log "Installing CUDA ${TORCH_CUDA} compiler + headers via pip (no root needed)"
  if pip install --quiet \
       "nvidia-cuda-nvcc-cu${TORCH_CUDA_MAJOR}==${TORCH_CUDA}.*" \
       "nvidia-cuda-runtime-cu${TORCH_CUDA_MAJOR}==${TORCH_CUDA}.*" \
       "nvidia-cuda-cccl-cu${TORCH_CUDA_MAJOR}==${TORCH_CUDA}.*" 2>/dev/null \
     || pip install --quiet \
       "nvidia-cuda-nvcc-cu${TORCH_CUDA_MAJOR}" \
       "nvidia-cuda-runtime-cu${TORCH_CUDA_MAJOR}"; then
    # Find the nvcc pip just installed. The layout varies between package
    # versions (nvidia/cuda_nvcc/bin, nvidia/cuda_nvcc/nvvm/..., or directly on
    # PATH via a console script), so search for the binary rather than assume
    # one location, then verify its release matches torch.
    _nvcc_bin="$(python - <<'PY' 2>/dev/null
import os, sys, glob, importlib.util as u
cands = []
s = u.find_spec("nvidia")
roots = list(s.submodule_search_locations) if (s and s.submodule_search_locations) else []
roots += [os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                       "site-packages", "nvidia")]
for r in roots:
    cands += glob.glob(os.path.join(r, "**", "bin", "nvcc"), recursive=True)
cands += glob.glob(os.path.join(sys.prefix, "bin", "nvcc"))
for c in cands:
    if os.path.isfile(c) and os.access(c, os.X_OK):
        print(c); break
PY
)"
    _nvcc_dir=""
    if [ -n "$_nvcc_bin" ]; then
      # CUDA_HOME must be the toolkit ROOT (the dir containing bin/), not bin.
      _nvcc_dir="$(dirname "$(dirname "$_nvcc_bin")")"
    fi
    if [ -n "$_nvcc_dir" ] && [ -x "$_nvcc_dir/bin/nvcc" ]; then
      # The pip wheels split the toolkit across sibling packages
      # (cuda_nvcc/bin, cuda_runtime/include, cuda_cccl/include). torch expects
      # ONE root with bin/ + include/ underneath, so stitch a toolkit together
      # out of symlinks in a writable dir.
      _shim="$WORKSPACE/.cuda_shim_${TORCH_CUDA}"
      rm -rf "$_shim"; mkdir -p "$_shim/bin" "$_shim/include" "$_shim/lib64"
      ln -sf "$_nvcc_dir"/bin/* "$_shim/bin/" 2>/dev/null || true
      [ -d "$_nvcc_dir/nvvm" ] && ln -sfn "$_nvcc_dir/nvvm" "$_shim/nvvm"
      _nvroot="$(dirname "$_nvcc_dir")"
      for pkg in cuda_runtime cuda_cccl cuda_nvcc; do
        [ -d "$_nvroot/$pkg/include" ] && ln -sf "$_nvroot/$pkg"/include/* "$_shim/include/" 2>/dev/null || true
        [ -d "$_nvroot/$pkg/lib" ]     && ln -sf "$_nvroot/$pkg"/lib/*     "$_shim/lib64/"   2>/dev/null || true
      done
      export CUDA_HOME="$_shim"
      export PATH="$_shim/bin:$PATH"
      _ver="$("$_shim/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
      log "  nvcc $_ver ready at $_shim"
      if [ ! -f "$_shim/include/cuda_runtime.h" ]; then
        warn "  cuda_runtime.h missing from the shim — the build may still fail."
        warn "  If it does: sudo apt-get install -y cuda-toolkit-${TORCH_CUDA/./-}"
      fi
    else
      warn "pip nvcc installed but binary not found; falling back to apt"
      warn "Run manually if the build fails:"
      warn "  sudo apt-get install -y cuda-toolkit-${TORCH_CUDA/./-}"
      SKIP_FRNN=1
    fi
  else
    warn "Could not install a matching nvcc."
    warn "Try:  sudo apt-get install -y cuda-toolkit-${TORCH_CUDA/./-}"
    warn "Or reinstall torch for the system CUDA:"
    warn "  TORCH_VERSION=2.5.1 bash 01_setup_env.sh   # then re-run this script"
    SKIP_FRNN=1
  fi
else
  export CUDA_HOME="$CUDA_MATCH"
  export PATH="$CUDA_MATCH/bin:$PATH"
fi

# The clone path is NOT free-form: src/utils/neighbors.py does
#   from src.dependencies.FRNN import frnn
# at module import time, so the tree must live exactly at
# src/dependencies/FRNN or every SPT import fails.
log "Building FRNN (compiles CUDA kernels, several minutes)"
if [ ! -d src/dependencies/FRNN ]; then
  git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
fi
# --no-build-isolation is REQUIRED here. FRNN's setup.py does `import torch` at
# build time to locate CUDA headers, but pip builds in an isolated venv where
# torch is absent -> "ModuleNotFoundError: No module named 'torch'". Disabling
# isolation lets setup.py see the torch we installed in 01_setup_env.sh.
# setuptools/wheel must then be present in the real env, since the isolated
# build env is what normally provides them.
pip install --quiet "setuptools<70" wheel ninja
if [ "$SKIP_FRNN" = "1" ]; then
  echo "  skipped (no matching CUDA toolkit)"
else
  # FRNN links against torch's C++/CUDA ABI, so a binary built for one torch
  # version fails on another with e.g.
  #   undefined symbol: _ZN3c104cuda9SetDeviceEi
  # pip would otherwise see it "already installed" and skip, leaving the stale
  # .so in place. Rebuild whenever the torch it was built against differs.
  _frnn_ok=0
  if python -c "from frnn import _C" >/dev/null 2>&1; then
    _frnn_ok=1
  fi
  if [ "$_frnn_ok" = "1" ]; then
    echo "  FRNN already built for this torch"
  else
    pip uninstall -y -q frnn prefix_sum 2>/dev/null || true
    # Stale build trees also cache objects from the previous torch.
    rm -rf src/dependencies/FRNN/build src/dependencies/FRNN/*.egg-info \
           src/dependencies/FRNN/external/prefix_sum/build 2>/dev/null || true
    (
      cd src/dependencies/FRNN/external/prefix_sum \
        && pip install --quiet --no-build-isolation --no-cache-dir .
      cd ../.. && pip install --quiet --no-build-isolation --no-cache-dir .
    ) && python -c "from frnn import _C" 2>/dev/null \
      && echo "  FRNN ok" \
      || warn "FRNN build failed or produced an unloadable extension"
  fi
fi

# ── 4. SPT's remaining dependencies ────────────────────────────────────────
# Taken from the package list inside the repo's own install.sh, but installed
# directly rather than by running it. That script is not usable here: it gates
# on the SYSTEM CUDA (nvcc) accepting only 11.8/12.1 while Lightning Studios
# ship CUDA 13.0, and it builds its own conda env. Neither matters for what we
# need -- the only CUDA that counts is the 12.1 bundled with the torch wheel.
#
# cut-pursuit and grid-graph are published as WHEELS (pycut-pursuit /
# pygrid-graph); there is nothing to compile from source for them.
log "SPT graph dependencies (from install.sh's package list)"
pip install --quiet \
  pycut-pursuit pygrid-graph pgeof \
  torch-graph-components torch-ransac3d \
  pyrootutils gdown \
  || warn "some SPT deps failed"

# torch_cluster / pyg_lib come from the version-matched wheel index, like
# torch-scatter in 01_setup_env.sh.
PYG_TAG="$(python -c 'import torch; print("torch-"+torch.__version__.split("+")[0]+"+cu121")')"
log "torch_cluster / pyg_lib (${PYG_TAG})"
# Same stale-binary trap as in 01_setup_env.sh: these link against a specific
# torch build, so force a clean fetch rather than letting pip skip them.
pip uninstall -y -q torch-cluster pyg-lib 2>/dev/null || true
pip install --quiet --no-cache-dir pyg_lib torch_cluster -f "https://data.pyg.org/whl/${PYG_TAG}.html" \
  || warn "torch_cluster/pyg_lib wheels unavailable for ${PYG_TAG}"

# Fail loudly here rather than at SPT import time — a symbol error deep in a
# .so is much harder to read than this check.
python - <<'PY' || warn "PyG extensions are broken - see the uninstall/reinstall note above"
import torch
mods = {}
for m in ("torch_scatter", "torch_sparse", "torch_cluster"):
    try:
        __import__(m); mods[m] = "ok"
    except Exception as e:
        mods[m] = f"BROKEN ({type(e).__name__})"
print(f"  torch {torch.__version__}")
for k, v in mods.items():
    print(f"  {k}: {v}")
raise SystemExit(0 if all(v == "ok" for v in mods.values()) else 1)
PY

# ── 5. Checkpoint ──────────────────────────────────────────────────────────
# The server loads ~/spt_dales.ckpt. It is NOT downloaded automatically:
# fetch the DALES checkpoint from the SPT model zoo and place it there.
CKPT="$HOME/spt_dales.ckpt"
if [ -f "$CKPT" ]; then
  log "Checkpoint present: $CKPT ($(du -h "$CKPT" | cut -f1))"
else
  warn "Missing checkpoint: $CKPT"
  warn "Download the DALES weights from the SPT model zoo (README) and save to that path."
  warn "The server will not start without it."
fi

# ── 6. Re-pin numpy ────────────────────────────────────────────────────────
# Several packages above (torch_geometric, numba, plotly, ...) declare
# "numpy" with no upper bound and will happily pull 2.x, which breaks the
# torch<->numpy bridge:
#   "A module compiled using NumPy 1.x cannot be run in NumPy 2.2.6"
#   "Failed to initialize NumPy: _ARRAY_API not found"
# torch 2.2.0 wheels are built against numpy 1.x, so this MUST be re-asserted
# after every install pass, not just once at the start.
log "Re-pinning numpy <2 (installs above may have upgraded it)"
_np="$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo none)"
if [ "${_np%%.*}" != "1" ]; then
  warn "numpy is $_np — reinstalling 1.26.4"
  # Any package still demanding numpy>=2 (plyfile 1.1+ did) must be downgraded
  # too, or pip reports a conflict and the next install silently re-upgrades
  # numpy. Pinning both together keeps the resolver consistent.
  pip install --quiet "numpy==1.26.4" "plyfile<1.1"
fi
python -c "import numpy, torch; print(f'  numpy {numpy.__version__}  torch {torch.__version__}')"

# ── 7. Verify ──────────────────────────────────────────────────────────────
log "Verifying imports"
python - <<'PY'
import sys, os
ws = "/teamspace/studios/this_studio/superpoint_transformer"
if not os.path.isdir(ws):
    ws = os.path.expanduser("~/superpoint_transformer")
sys.path.insert(0, ws); os.chdir(ws)
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
for m in ("scipy.spatial", "laspy", "hydra", "omegaconf",
          "pytorch_lightning", "fastapi", "uvicorn"):
    try:
        __import__(m); print(f"  {m} ok")
    except Exception as e:
        print(f"  [FAIL] {m}: {e}")
try:
    from src.models.semantic import SemanticSegmentationModule  # noqa: F401
    from src.transforms import instantiate_transforms           # noqa: F401
    print("  SPT src.* ok")
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "") or str(e)
    if "FRNN" in missing:
        # SPT imports FRNN from src/dependencies/FRNN. Without the compiled
        # extension that import fails outright, so the clone must at least be
        # present at that path even when the build was skipped.
        print(f"  [FAIL] SPT src.*: {e}")
        print("         FRNN was not built (CUDA mismatch). SPT imports it")
        print("         unconditionally, so inference will not start without it.")
        print("         Options: install a CUDA 12.1 toolkit, or use a Studio")
        print("         image whose system CUDA matches torch.")
    else:
        print(f"  [FAIL] SPT src.*: {e}")
        print("         A python dependency is missing — check step 2/4 output.")
except Exception as e:
    print(f"  [FAIL] SPT src.*: {type(e).__name__}: {e}")
PY

log "Done"
echo "Start the server with:"
echo "  uvicorn spt_server:app --host 0.0.0.0 --port 8000"
