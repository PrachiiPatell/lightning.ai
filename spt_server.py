"""SPT-DALES inference server matching JLabel's /predict contract."""
import os, sys, time, tempfile, threading, urllib.request
import numpy as np, torch, laspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Locate the superpoint_transformer checkout. 03_install_spt.sh clones it into
# the Lightning Studio workspace when present and into $HOME otherwise (e.g. on
# a plain EC2 box), so resolve at run time instead of hardcoding one host's
# layout. Override with SPT_ROOT=/path/to/superpoint_transformer.
_SPT_ROOT = os.environ.get("SPT_ROOT") or next(
    (p for p in (
        "/teamspace/studios/this_studio/superpoint_transformer",
        os.path.expanduser("~/superpoint_transformer"),
    ) if os.path.isdir(p)),
    None,
)
if not _SPT_ROOT:
    sys.exit(
        "superpoint_transformer checkout not found. Run 03_install_spt.sh first, "
        "or set SPT_ROOT to the checkout path."
    )
sys.path.insert(0, _SPT_ROOT)
os.chdir(_SPT_ROOT)
torch.backends.cuda.preferred_linalg_library("magma")

from omegaconf import OmegaConf
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
import hydra
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

# DALES class index -> JLabel semantic overlay index
DALES_TO_JLABEL_SEM = {
    0: 1,   # Ground -> Ground
    1: 4,   # Vegetation -> Trees
    2: 2,   # Cars -> Cars
    3: 3,   # Trucks -> Trucks
    4: 7,   # Power lines -> Power lines
    5: 8,   # Fences -> Fences
    6: 6,   # Poles -> Poles
    7: 5,   # Buildings -> Buildings
}
DALES_NAMES = ["Ground", "Vegetation", "Cars", "Trucks",
               "Power lines", "Fences", "Poles", "Buildings"]

CLASS_ALIASES = {
    "Cars":        ["car", "vehicle", "auto"],
    "Trucks":      ["truck", "lorry", "van"],
    "Trees":       ["tree", "vegetation_tree", "tall_vegetation"],
    "Buildings":   ["building", "house", "structure"],
    "Poles":       ["pole", "light_pole", "traffic_sign", "sign"],
    "Power lines": ["power line", "powerline", "wire", "cable", "conductor"],
    "Fences":      ["fence", "railing", "barrier"],
}


def _resolve_label(spt_class, user_classes):
    candidates = CLASS_ALIASES.get(spt_class, [spt_class.lower().rstrip("s")])
    for cand in candidates:
        for u in user_classes:
            if u.lower() == cand:
                return u
    return None


print("Loading SPT-DALES model...", flush=True)
hydra.core.global_hydra.GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["experiment=semantic/dales"])

from src.models.semantic import SemanticSegmentationModule
from src.data import Data
from src.transforms import instantiate_transforms

_net = instantiate(cfg.model.net)
_criterion = instantiate(cfg.model.criterion)
MODEL = SemanticSegmentationModule(
    net=_net, criterion=_criterion,
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.ConstantLR,
    num_classes=cfg.datamodule.num_classes,
    multi_stage_loss_lambdas=[1, 50],
)
_ck = torch.load(os.path.expanduser("~/spt_dales.ckpt"),
                 map_location="cpu", weights_only=False)
MODEL.load_state_dict(_ck["state_dict"], strict=False)
MODEL = MODEL.to("cuda").eval()
PRE_T = instantiate_transforms(cfg.datamodule.pre_transform)
TEST_T = instantiate_transforms(cfg.datamodule.on_device_test_transform)
print("Model ready.", flush=True)

app = FastAPI()
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
inference_lock = threading.Lock()


class AiAssistRequest(BaseModel):
    lidar_file_url: str
    class_colors: dict = {}
    model: str | None = None
    score_threshold: float = 0.0
    calib_file_url: str | None = None

    class Config:
        extra = "allow"


@app.get("/health")
def health():
    return {"ok": True, "model": "SPT-DALES"}


@app.post("/predict")
def predict(req: AiAssistRequest):
    t0 = time.time()
    ext = os.path.splitext(req.lidar_file_url.split("?")[0])[1].lower()
    if ext not in (".las", ".laz"):
        raise HTTPException(400, f"SPT-DALES only supports .las/.laz, got {ext}")
    print(f"[REQ] {req.lidar_file_url[:80]}", flush=True)

    with inference_lock:
        # 1. Download the file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for attempt in range(3):
                try:
                    tmp.seek(0); tmp.truncate(0)
                    with urllib.request.urlopen(req.lidar_file_url, timeout=180) as r:
                        while True:
                            chunk = r.read(4 * 1024 * 1024)
                            if not chunk: break
                            tmp.write(chunk)
                    break
                except Exception as e:
                    print(f"[DL retry {attempt+1}/3] {type(e).__name__}: {e}", flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                raise HTTPException(502, "Download failed after 3 attempts")
            tmp_path = tmp.name
        print(f"[DL] done in {time.time()-t0:.1f}s", flush=True)

        try:
            # 2. Read LAS as float32 with a hard memory cap
            t = time.time()
            HARD_CAP = 3_000_000
            with laspy.open(tmp_path) as _f:
                _n_total = _f.header.point_count
            las = laspy.read(tmp_path)
            # Original per-point RGB — powers single-cloud erase/hide in JLabel
            # (erased points repaint to their true colour in place).
            _dims = set(las.point_format.dimension_names)
            _has_rgb = {'red', 'green', 'blue'} <= _dims
            if _n_total > HARD_CAP:
                _sel = np.random.default_rng(42).choice(_n_total, HARD_CAP, replace=False)
                _sel.sort()
                xyz_all = np.stack([
                    np.asarray(las.x, dtype=np.float32)[_sel],
                    np.asarray(las.y, dtype=np.float32)[_sel],
                    np.asarray(las.z, dtype=np.float32)[_sel],
                ], axis=-1)
                intensity_all = np.asarray(las.intensity, dtype=np.float32)[_sel]
                rgb_all = np.stack([
                    np.asarray(las.red, dtype=np.float32)[_sel],
                    np.asarray(las.green, dtype=np.float32)[_sel],
                    np.asarray(las.blue, dtype=np.float32)[_sel],
                ], axis=-1) if _has_rgb else None
                del las
                import gc; gc.collect()
            else:
                xyz_all = np.stack([
                    np.asarray(las.x, dtype=np.float32),
                    np.asarray(las.y, dtype=np.float32),
                    np.asarray(las.z, dtype=np.float32),
                ], axis=-1)
                intensity_all = np.asarray(las.intensity, dtype=np.float32)
                rgb_all = np.stack([
                    np.asarray(las.red, dtype=np.float32),
                    np.asarray(las.green, dtype=np.float32),
                    np.asarray(las.blue, dtype=np.float32),
                ], axis=-1) if _has_rgb else None
            if rgb_all is not None:
                if rgb_all.max() > 255:      # LAS colour is usually 16-bit
                    rgb_all = rgb_all / 257.0
                rgb_all = np.clip(rgb_all, 0, 255).astype(np.uint8)
            n = len(xyz_all)
            centroid = xyz_all.mean(axis=0)
            print(f"[READ] {n:,} pts, rgb={_has_rgb}, {time.time()-t:.1f}s", flush=True)

            # 3. Drop vertical outliers BEFORE subsampling.
            #
            # SPT's preprocessing fits a ground plane with RANSAC
            # (src/utils/ground.py). A handful of stray returns far below the
            # terrain skew that fit badly enough that it finds no plane at all
            # and returns None -- which the library then dereferences, crashing
            # the request with a 500. One real 1 km tile had p0=12.1 m against
            # p1=65.5 m: a 53 m gap made of noise.
            #
            # Clip to the 0.5-99.5 percentile band. That removes the stragglers
            # while keeping every real surface, since genuine terrain relief is
            # continuous rather than a lone spike.
            _zlo, _zhi = np.percentile(xyz_all[:, 2], [0.5, 99.5])
            _keep = (xyz_all[:, 2] >= _zlo) & (xyz_all[:, 2] <= _zhi)
            _dropped = int(n - _keep.sum())
            if _dropped:
                xyz_all = xyz_all[_keep]
                intensity_all = intensity_all[_keep]
                if rgb_all is not None:
                    rgb_all = rgb_all[_keep]
                n = len(xyz_all)
                centroid = xyz_all.mean(axis=0)
                print(f"[CLIP] dropped {_dropped:,} z-outliers "
                      f"(kept {_zlo:.1f}..{_zhi:.1f} m), {n:,} pts left", flush=True)

            # 4. Subsample for SPT inference.
            #
            # Raised from 1.5M: an 80M-point 1 km tile subsampled to 1.5M leaves
            # ~1.5 pts/m2, too sparse for the ground fit to lock on. The L4 has
            # 23 GB VRAM and handles 4M comfortably.
            MAX_PTS = int(os.environ.get("SPT_MAX_PTS", "4000000"))
            if n > MAX_PTS:
                idx = np.random.default_rng(42).choice(n, MAX_PTS, replace=False)
                idx.sort()
            else:
                idx = np.arange(n)
            pos = xyz_all[idx] - centroid
            intensity = intensity_all[idx]

            # 4. SPT forward pass (GPU)
            t = time.time()
            data = Data()
            data.pos = torch.from_numpy(pos)
            data.intensity = torch.from_numpy(np.clip(intensity, 0, 60000) / 60000).unsqueeze(-1)
            # SPT's ground-plane RANSAC returns None when it cannot fit a plane,
            # and src/utils/ground.py dereferences that without checking
            # ("TypeError: 'NoneType' object is not subscriptable"). Translate
            # it into an answer the caller can act on instead of a bare 500.
            try:
                nag = PRE_T(data)
            except TypeError as _e:
                if "NoneType" in str(_e):
                    raise HTTPException(
                        422,
                        "Could not detect a ground plane in this cloud. SPT-DALES "
                        "expects an aerial survey with visible terrain; a scan "
                        "that is very sparse, very steep, or has no ground "
                        "returns will fail here.",
                    )
                raise
            nag = TEST_T(nag.to("cuda"))
            with torch.no_grad():
                _, output = MODEL.predict_step(nag, batch_idx=0)
            print(f"[SPT] {time.time()-t:.1f}s", flush=True)

            sp_preds = output.semantic_pred().cpu().numpy()
            super_index = nag[0].super_index.cpu().numpy()
            voxel_preds = sp_preds[super_index]
            voxel_pos = nag[0].pos.cpu().numpy() + centroid

            # 5. Project voxel predictions to all original points
            t = time.time()
            from scipy.spatial import cKDTree
            tree = cKDTree(voxel_pos)
            _, nn = tree.query(xyz_all, k=1, workers=-1)
            per_point = voxel_preds[nn].astype(np.int32)
            print(f"[NN] {time.time()-t:.1f}s", flush=True)

            # 6. Semantic overlay — PER-POINT: every point read from the LAS gets
            # its predicted class, no subsampling. Coords rounded to cm and the
            # response is gzipped, so a full 2-3M-point overlay stays manageable.
            keep = np.arange(len(xyz_all))
            ov_pos = np.round(xyz_all[keep] - centroid, 2)
            # Vectorized DALES->JLabel class map (fast for millions of points).
            _dales_map = np.array(
                [DALES_TO_JLABEL_SEM[i] for i in range(len(DALES_NAMES))],
                dtype=np.int32,
            )
            ov_cls = _dales_map[per_point[keep]]
            ov_rgb = rgb_all[keep] if rgb_all is not None else None

            print(f"[OVERLAY] per-point: {len(keep):,} pts, rgb={ov_rgb is not None}", flush=True)
            _t_ser = time.time()

            # .tolist() builds tens of millions of Python objects and the
            # JSON encode that follows is comparable again -- measured at
            # ~24s / 216 MB for 5M points. Logged so a slow request can be
            # attributed to serialisation rather than blamed on the model.
            semantic_overlay = {
                "positions": ov_pos.tolist(),
                "classes": ov_cls.tolist(),
                "rgb": ov_rgb.tolist() if ov_rgb is not None else None,
                "instance_ids": [0] * len(ov_cls),
                "palette": {
                    "Other":       [80, 80, 80],
                    "Ground":      [243, 214, 171],
                    "Cars":        [233, 50, 239],
                    "Trucks":      [243, 238, 0],
                    "Trees":       [70, 115, 66],
                    "Buildings":   [214, 66, 54],
                    "Poles":       [239, 114, 0],
                    "Power lines": [190, 153, 153],
                    "Fences":      [0, 233, 11],
                },
            }
            print(f"[SERIALISE] {time.time()-_t_ser:.1f}s "
                  f"({len(semantic_overlay['positions']):,} pts to Python lists)",
                  flush=True)

            # 7. Extra classes + per-class point counts
            user_classes = list(req.class_colors.keys())
            extra_classes = []
            pending_cuboids = {}
            class_counts = {}
            detected_dales = set(int(c) for c in np.unique(per_point))
            print(f"[EXTRAS] detected DALES classes: {sorted(detected_dales)}", flush=True)
            for dales_idx in detected_dales:
                if not (0 <= dales_idx < len(DALES_NAMES)):
                    continue
                name = DALES_NAMES[dales_idx]
                if name == "Vegetation":
                    name = "Trees"
                class_counts[name] = int((per_point == dales_idx).sum())
                if _resolve_label(name, user_classes):
                    continue
                if name not in extra_classes:
                    extra_classes.append(name)
                    pending_cuboids[name] = []
            print(f"[EXTRAS] returning: {extra_classes}", flush=True)

        finally:
            os.unlink(tmp_path)

    elapsed = time.time() - t0
    print(f"[RESP] 0 cuboids, {len(extra_classes)} extra_classes, TOTAL {elapsed:.1f}s",
          flush=True)
    return {
        "predictions": [],
        "semantic_overlay": semantic_overlay,
        "extra_classes": extra_classes,
        "pending_cuboids": pending_cuboids,
        "class_counts": class_counts,
    }


if __name__ == "__main__":
    # Allow `python spt_server.py` as well as `uvicorn spt_server:app ...`.
    # The systemd unit in DEPLOY_EC2.md invokes the module directly, and
    # without this the process would load the model, define `app`, then exit
    # immediately -- systemd would read that as a crash-loop.
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("SPT_HOST", "0.0.0.0"),
        port=int(os.environ.get("SPT_PORT", "8000")),
    )
