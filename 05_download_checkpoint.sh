#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch the SPT-DALES pretrained weights.
#
# Source: Zenodo record 8042712, "Efficient 3D Semantic Segmentation with
# Superpoint Transformer" -- the authors' official release, linked from the
# repo README. DALES is the AERIAL dataset, which is why this is the right
# checkpoint for airborne LAS tiles (the S3DIS/KITTI-360 weights in the same
# record are indoor and automotive respectively).
#
# The inference server does torch.load("~/spt_dales.ckpt"), so the file is
# saved under that exact name regardless of what Zenodo calls it.
#
#   bash 05_download_checkpoint.sh
# ---------------------------------------------------------------------------
set -euo pipefail

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }

ZENODO_FILE="spt-2_dales.ckpt"
URL="https://zenodo.org/api/records/8042712/files/${ZENODO_FILE}/content"
DEST="$HOME/spt_dales.ckpt"

if [ -f "$DEST" ]; then
  log "Already present: $DEST ($(du -h "$DEST" | cut -f1))"
  echo "Delete it first if you want to re-download."
  exit 0
fi

log "Downloading ${ZENODO_FILE} (~2.9 MB)"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$DEST.part" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "$DEST.part" "$URL"
else
  warn "Neither curl nor wget found."
  exit 1
fi

# Only move into place after a successful download, so an interrupted transfer
# never leaves a truncated file that torch.load would fail on cryptically.
mv "$DEST.part" "$DEST"
log "Saved to $DEST ($(du -h "$DEST" | cut -f1))"

log "Verifying it loads"
# Must run from the SPT repo: the checkpoint is a Lightning pickle referencing
# SPT's own classes, so `src` has to be importable or unpickling raises
# ModuleNotFoundError even though the file is perfectly valid.
_SPT="/teamspace/studios/this_studio/superpoint_transformer"
[ -d "$_SPT" ] || _SPT="$HOME/superpoint_transformer"
cd "$_SPT" 2>/dev/null || warn "SPT repo not found — verification may fail spuriously"
python - <<'PY'
import os, sys, torch
sys.path.insert(0, os.getcwd())
p = os.path.expanduser("~/spt_dales.ckpt")
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    raise SystemExit(1)
sd = ck.get("state_dict", ck)
print(f"  ok - {len(sd)} tensors")
ep = ck.get("epoch")
if ep is not None:
    print(f"  epoch: {ep}")
PY

echo
echo "Start the server with:"
echo "  uvicorn spt_server:app --host 0.0.0.0 --port 8000"
