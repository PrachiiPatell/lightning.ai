"""SPT-DALES inference server matching JLabel's /predict contract."""
import os, sys, time, base64, tempfile, threading, urllib.request
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


# ---------------------------------------------------------------------------
# PLY reading.
#
# SPT-DALES itself is format-agnostic -- everything past the read step is plain
# numpy. Only laspy was, so .ply was rejected at the door even though JLabel
# routes it here. These two functions are ported from JLabel's own PLY parser
# (main/views.py) rather than adding a plyfile dependency: that parser is
# already proven against the files real users upload, and a second independent
# implementation could disagree with it on the same file.
#
# The result mimics just enough of a laspy object for the read step below:
# .x/.y/.z, .intensity, and optional .red/.green/.blue.
# ---------------------------------------------------------------------------
_PLY_TYPE_MAP = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


class _PlyPoints:
    """Duck-types the subset of a laspy object the read step touches."""
    pass


def _parse_ply_header(f):
    """Read the ASCII header of a (possibly binary-body) PLY.

    Returns (format, elements), elements being an ordered list of
    {'name', 'count', 'props': [(np_dtype_str, prop_name), ...]}. Only
    fixed-size properties are supported, which is correct for vertex data.
    """
    line = f.readline().decode('ascii').strip()
    if line != 'ply':
        raise ValueError('Not a PLY file')
    fmt = None
    elements = []
    while True:
        line = f.readline().decode('ascii').strip()
        if line.startswith('format'):
            fmt = line.split()[1]
        elif line.startswith('element'):
            _, name, count = line.split()
            elements.append({'name': name, 'count': int(count), 'props': []})
        elif line.startswith('property'):
            parts = line.split()
            if parts[1] == 'list':
                raise ValueError(
                    f'PLY list properties not supported (element: {elements[-1]["name"]})')
            elements[-1]['props'].append((_PLY_TYPE_MAP[parts[1]], parts[2]))
        elif line == 'end_header':
            break
        elif line == '':
            raise ValueError('Unexpected EOF in PLY header')
    return fmt, elements


def _read_ply(path):
    """Parse a PLY into a laspy-like object. Returns (obj, n_points)."""
    with open(path, 'rb') as f:
        fmt, elements = _parse_ply_header(f)
        vertex_el = next((e for e in elements if e['name'] == 'vertex'), None)
        if vertex_el is None:
            raise ValueError('PLY has no vertex element')
        names = [p[1] for p in vertex_el['props']]
        for req in ('x', 'y', 'z'):
            if req not in names:
                raise ValueError(f'PLY vertex element missing "{req}" property')

        if fmt == 'ascii':
            # Elements listed before 'vertex' must be consumed line by line.
            for el in elements:
                if el is vertex_el:
                    break
                for _ in range(el['count']):
                    f.readline()
            cols = {n: [] for n in names}
            for _ in range(vertex_el['count']):
                vals = f.readline().decode('ascii').split()
                for n, v in zip(names, vals):
                    cols[n].append(float(v))
            arrs = {n: np.asarray(v, dtype=np.float32) for n, v in cols.items()}
        else:
            endian = '<' if fmt == 'binary_little_endian' else '>'
            body_offset = 0
            for el in elements:
                if el is vertex_el:
                    break
                dt = np.dtype([(n, endian + t) for t, n in el['props']])
                body_offset += dt.itemsize * el['count']
            dt = np.dtype([(n, endian + t) for t, n in vertex_el['props']])
            f.seek(body_offset, 1)   # relative: we are just past end_header
            raw = f.read(dt.itemsize * vertex_el['count'])
            struct_arr = np.frombuffer(raw, dtype=dt)
            arrs = {n: np.asarray(struct_arr[n], dtype=np.float32) for n in names}

    n = len(arrs['x'])
    obj = _PlyPoints()
    obj.x, obj.y, obj.z = arrs['x'], arrs['y'], arrs['z']

    # Intensity: PLY rarely carries it. DALES trained WITH intensity, so a
    # constant fill is a real information loss -- expect somewhat weaker
    # labels on PLY than on LAS. Accept the common spellings before falling
    # back. The fill is 1.0, not 0.0, to match the neutral value the LAS path
    # produces for files whose intensity dimension is present but unwritten.
    for _cand in ('intensity', 'scalar_Intensity', 'scalar_intensity'):
        if _cand in arrs:
            obj.intensity = arrs[_cand]
            break
    else:
        obj.intensity = np.ones(n, dtype=np.float32)

    # PLY colour is uchar 0-255 already. The LAS path divides by 257 when it
    # sees values above 255 (16-bit LAS colour); leaving these 8-bit means
    # that branch correctly does nothing.
    if all(c in arrs for c in ('red', 'green', 'blue')):
        obj.red, obj.green, obj.blue = arrs['red'], arrs['green'], arrs['blue']
    return obj, n


@app.get("/health")
def health():
    return {"ok": True, "model": "SPT-DALES"}


@app.post("/predict")
def predict(req: AiAssistRequest):
    t0 = time.time()
    ext = os.path.splitext(req.lidar_file_url.split("?")[0])[1].lower()
    if ext not in (".las", ".laz", ".ply"):
        raise HTTPException(400, f"SPT-DALES only supports .las/.laz/.ply, got {ext}")
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
            # 2. Read LAS as float32, bounded by SPT_HARD_CAP.
            #
            # There are TWO caps in this handler and they bound different
            # resources for different reasons. Keep them separate:
            #
            #   SPT_MAX_PTS   how many points the MODEL sees (step 4).
            #                 Bounded by VRAM. Governs prediction QUALITY.
            #   SPT_HARD_CAP  how many points get LABELLED and returned (here).
            #                 Bounded by host RAM. Governs output RESOLUTION.
            #
            # They are decoupled, which is easy to miss: SPT predicts on
            # superpoints, those are projected to voxels, and step 5 then
            # labels points with a KDTree lookup against the voxel positions.
            # That lookup does not care how many points the model saw -- so we
            # can infer on 4M and still label all 20M. Sending more points to
            # the GPU is expensive; labelling more points is a KDTree query and
            # some RAM.
            #
            # This cap does NOT protect the read itself. laspy.read() below
            # loads the whole file before we get here, so that peak is paid
            # either way; what the cap bounds is the arrays we RETAIN (xyz,
            # intensity, rgb) and the size of the step 5 query. At float32 that
            # is ~19 bytes/point retained after rgb is packed to uint8, so a
            # 12.5M-point tile costs ~240 MB and the 20M ceiling ~380 MB. The
            # ceiling exists for the pathological upload -- an 80M-point 1 km
            # tile would retain ~1.5 GB on top of laspy's own read -- not for
            # ordinary tiles, which should never hit it.
            #
            # Previously this was hardcoded at 3M while SPT_MAX_PTS was 4M,
            # which had two bad effects. It silently made the step 4 subsample
            # dead code (`n` was already <= 3M, so `n > MAX_PTS` never fired,
            # and the documented 1.5M -> 4M raise never took effect). And it
            # capped OUTPUT resolution at 3M for no gain: a 12.5M-point
            # DALES-sized tile came back labelled at 9.3 pts/m2 against the
            # ~50 pts/m2 the model trains on, having discarded 76% of the
            # points to save ~180 MB.
            #
            # The max() is a guard against that first failure returning by
            # misconfiguration, not a coupling -- HARD_CAP should normally sit
            # far ABOVE MAX_PTS, not equal to it.
            t = time.time()
            MAX_PTS = int(os.environ.get("SPT_MAX_PTS", "4000000"))
            HARD_CAP = max(int(os.environ.get("SPT_HARD_CAP", "20000000")),
                           MAX_PTS)
            if ext == ".ply":
                # No cheap header-only count: the PLY parser reads the whole
                # vertex block in one pass, so the count comes back with it.
                las, _n_total = _read_ply(tmp_path)
                _has_rgb = all(hasattr(las, c) for c in ('red', 'green', 'blue'))
            else:
                with laspy.open(tmp_path) as _f:
                    _n_total = _f.header.point_count
                las = laspy.read(tmp_path)
                # Original per-point RGB — powers single-cloud erase/hide in
                # JLabel (erased points repaint to their true colour in place).
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
                # Which SOURCE point each row came from. Carried through the
                # z-clip below and used in step 6 to return per-point classes
                # indexed to the file the caller has, not to our working array.
                src_idx = _sel
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
                src_idx = np.arange(len(xyz_all), dtype=np.int64)
            # Free the laspy object on BOTH paths. It is not referenced again,
            # and everything below works off xyz_all/intensity_all/rgb_all.
            #
            # This used to sit inside the capped branch only, which was
            # survivable while the cap was 3M (nearly every real file took that
            # branch). With the ceiling at 20M almost everything takes the
            # uncapped branch instead, so leaving it there would keep the full
            # laspy object alive alongside our own arrays for the whole
            # request -- roughly doubling the read's footprint at exactly the
            # sizes the higher ceiling is meant to let through.
            del las
            import gc; gc.collect()
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
            # Percentiles were the wrong tool. On a dense urban tile most
            # points sit near ground level, so the 99.5th percentile lands
            # BELOW the rooftops: one real tile kept only 5.9-24.9 m out of
            # a 72 m scene, discarding every building top and treetop before
            # inference. The overlay then covered a thin slab and only ~25%
            # of points ended up classified on export.
            #
            # Clip by ABSOLUTE distance from the median instead. Genuine
            # structure lies within tens of metres of the ground; the noise
            # this guards against sat ~53 m below it. A generous band keeps
            # all real geometry while still cutting the stragglers that make
            # the ground-plane RANSAC fail.
            _zmed = float(np.median(xyz_all[:, 2]))
            _zmad = float(np.median(np.abs(xyz_all[:, 2] - _zmed))) or 1.0
            # 40 MADs is very wide -- it is a noise gate, not a crop.
            _band = max(40.0 * _zmad, 150.0)
            _zlo, _zhi = _zmed - _band, _zmed + _band
            _keep = (xyz_all[:, 2] >= _zlo) & (xyz_all[:, 2] <= _zhi)
            _dropped = int(n - _keep.sum())
            if _dropped:
                xyz_all = xyz_all[_keep]
                intensity_all = intensity_all[_keep]
                if rgb_all is not None:
                    rgb_all = rgb_all[_keep]
                # Keep the source mapping in step with the rows it describes;
                # step 6 scatters predictions back through it. Dropping this
                # would silently shift every per-point class by however many
                # outliers were removed.
                src_idx = src_idx[_keep]
                n = len(xyz_all)
                centroid = xyz_all.mean(axis=0)
                print(f"[CLIP] dropped {_dropped:,} z-outliers "
                      f"(kept {_zlo:.1f}..{_zhi:.1f} m), {n:,} pts left", flush=True)

            # 4. Subsample for SPT inference.
            #
            # Raised from 1.5M: an 80M-point 1 km tile subsampled to 1.5M leaves
            # ~1.5 pts/m2, too sparse for the ground fit to lock on. The L4 has
            # 23 GB VRAM and handles 4M comfortably.
            #
            # MAX_PTS is read in step 2, where HARD_CAP is clamped to be at
            # least this value. Do NOT re-read it here: the read cap is applied
            # first, so the two have to be decided together or this branch goes
            # dead (see the note there).
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

            # 6. Semantic overlay.
            #
            # Capped, NOT per-point. Sending every point meant ~127 MB of
            # JSON (~25 MB gzipped) for a 2.1M-point tile: Django parses it,
            # re-encodes it, ships it, and the browser parses it again --
            # none of which is visible in this log, which is why inference
            # looked fast while the overlay took far longer to appear.
            #
            # 500k matches what JLabel PERSISTS anyway (_persist_ai_overlay
            # downsamples to the same target), so the extra points were
            # discarded on the first save regardless -- they only ever cost
            # transfer time. Override with SPT_OVERLAY_MAX_PTS.
            _ov_cap = int(os.environ.get("SPT_OVERLAY_MAX_PTS", "500000"))
            _n_all = len(xyz_all)
            if _n_all > _ov_cap:
                # RANDOM sample rather than every Nth point.
                #
                # np.arange(0, n, step) selects by position in the array, which
                # here follows the order the LAS file stored the points -- i.e.
                # acquisition order. That makes the returned subset depend on
                # the scan pattern rather than on the scene, and it changes
                # entirely with the step size, so raising the cap silently
                # swapped which points the client saw.
                #
                # Sorted afterwards so the payload keeps its spatial locality
                # (and gzip ratio). Seeded for reproducibility.
                keep = np.sort(np.random.default_rng(12345).choice(
                    _n_all, size=_ov_cap, replace=False))
            else:
                keep = np.arange(_n_all)
            ov_pos = np.round(xyz_all[keep] - centroid, 2)
            # Vectorized DALES->JLabel class map (fast for millions of points).
            _dales_map = np.array(
                [DALES_TO_JLABEL_SEM[i] for i in range(len(DALES_NAMES))],
                dtype=np.int32,
            )
            ov_cls = _dales_map[per_point[keep]]
            ov_rgb = rgb_all[keep] if rgb_all is not None else None

            # 6b. FULL-RESOLUTION per-point classes, in SOURCE FILE ORDER.
            #
            # The sampled overlay above is ~4% of a 12.5M-point tile, and the
            # export rebuilds the missing 96% with a nearest-neighbour
            # propagation that amplifies every wrong sample ~25x (JLabel's
            # suppress_implausible_objects records 315 Pole samples becoming
            # 7,800 points that were 88.9% tree canopy). Returning the real
            # per-point answer removes that step rather than tuning it.
            #
            # It is also SMALLER than the sample it replaces. Measured on
            # M-34-63-A-b-2-4-4-1 (12,571,951 pts), gzipped as GZipMiddleware
            # already does:
            #
            #     500k positions + classes (the overlay)   3.67 MB    4% of pts
            #     per-class index runs (RLE), 100%         0.72 MB  100% of pts
            #     uint8 per point + base64, 100%           0.27 MB  100% of pts
            #
            # uint8 wins because JSON spends ~7 bytes on every integer while a
            # byte array spends one, and gzip handles the repetition better
            # than RLE-in-JSON does. It also degrades more gracefully: these
            # numbers come from a smoothed export, and rawer predictions mean
            # more runs, which costs RLE far more than it costs gzip.
            #
            # Indexed to the SOURCE file, NOT to xyz_all: the SPT_HARD_CAP
            # subsample and the z-outlier clip both drop rows, so src_idx is
            # carried through both and used to scatter back here. Points that
            # were dropped keep class 0, which JLabel already treats as
            # unclassified -- so the array is always exactly as long as the
            # caller's file, and alignment holds by construction rather than
            # by assumption.
            #
            # The sampled overlay is still returned alongside this: the viewer
            # renders from it, and older JLabel builds know nothing about
            # classes_b64.
            _t_pp = time.time()
            per_point_src = np.zeros(_n_total, dtype=np.uint8)
            per_point_src[src_idx] = _dales_map[per_point].astype(np.uint8)
            _pp_b64 = base64.b64encode(per_point_src.tobytes()).decode()
            print(f"[PERPOINT] {_n_total:,} pts, "
                  f"{len(_pp_b64)/1e6:.1f} MB b64 pre-gzip, "
                  f"{time.time()-_t_pp:.1f}s", flush=True)

            print(f"[OVERLAY] {len(keep):,} of {_n_all:,} pts "
                  f"(cap {_ov_cap:,}), rgb={ov_rgb is not None}", flush=True)
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
                # Full-resolution per-point classes (see 6b). uint8 per point,
                # base64, one entry per point of the SOURCE file in file order,
                # so a client can index it directly against its own LAS with no
                # spatial join. `n_points` is what that length must equal --
                # check it before trusting the alignment.
                "classes_b64": _pp_b64,
                "n_points": int(_n_total),
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
            # Per-class counts on the INFERRED points and on the OVERLAY that
            # is actually returned. If ground is plentiful here but scarce in
            # the export, the loss is downstream (persist cap / propagation),
            # not the model.
            _tot = len(per_point)
            _bits = []
            for _c in sorted(detected_dales):
                _n = int((per_point == _c).sum())
                _bits.append(f"{DALES_NAMES[_c]}={_n:,}({_n/_tot*100:.1f}%)")
            print("[COUNTS] inferred: " + "  ".join(_bits), flush=True)
            _ovtot = len(ov_cls)
            _ob = []
            for _c in sorted(detected_dales):
                _jl = DALES_TO_JLABEL_SEM[_c]
                _n = int((ov_cls == _jl).sum())
                _ob.append(f"{DALES_NAMES[_c]}={_n:,}({_n/_ovtot*100:.1f}%)")
            print("[COUNTS] overlay : " + "  ".join(_ob), flush=True)
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
