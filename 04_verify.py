#!/usr/bin/env python
"""Verify the Studio environment before running anything expensive.

Checks the things that actually break in this stack: version skew between
torch / spconv / torch-scatter, numpy 2.x silently breaking OpenPCDet, and a
GPU that is present but not usable from torch. Prints a single PASS/FAIL
summary so a broken env is obvious immediately rather than 20 minutes into a
training run.

    python 04_verify.py
"""

import sys

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

results = []


def check(name, fn, critical=True):
    try:
        detail = fn()
        results.append((True, name, detail, critical))
        print(f"{GREEN}  PASS{RESET}  {name:28} {detail}")
        return True
    except Exception as exc:
        results.append((False, name, str(exc), critical))
        tag = f"{RED}  FAIL{RESET}" if critical else f"{YELLOW}  WARN{RESET}"
        print(f"{tag}  {name:28} {exc}")
        return False


def c_python():
    v = sys.version_info
    if v < (3, 8):
        raise RuntimeError(f"python {v.major}.{v.minor} too old")
    return f"{v.major}.{v.minor}.{v.micro}"


def c_numpy():
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        raise RuntimeError(
            f"numpy {np.__version__} - OpenPCDet/spconv/open3d need <2.0; "
            "run: pip install 'numpy==1.26.4'")
    return np.__version__


def c_torch():
    import torch
    return f"{torch.__version__} (cuda {torch.version.cuda})"


def c_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False - "
                           "is this Studio on a GPU machine?")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    return f"{name}  {total:.0f}GB  sm_{cap}"


def c_gpu_compute():
    """A GPU that reports available can still fail on the first real kernel."""
    import torch
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    (a @ b).sum().item()
    torch.cuda.synchronize()
    return "matmul on device ok"


def c_spconv():
    import spconv
    return spconv.__version__


def c_scatter():
    import torch_scatter
    return getattr(torch_scatter, "__version__", "installed")


def c_laspy():
    import laspy
    backends = [str(b).split(".")[-1] for b in laspy.LazBackend.detect_available()]
    if not backends:
        raise RuntimeError("no LAZ backend - run: pip install 'laspy[lazrs]'")
    return f"{laspy.__version__}  LAZ: {','.join(backends)}"


def c_open3d():
    import open3d
    return open3d.__version__


def c_pcdet():
    import pcdet
    return pcdet.__version__


def c_scipy():
    import scipy
    return scipy.__version__


print("\nEnvironment verification")
print("=" * 60)

check("python", c_python)
check("numpy", c_numpy)
check("scipy", c_scipy, critical=False)
check("torch", c_torch)
check("cuda available", c_cuda)
check("gpu compute", c_gpu_compute)
check("laspy + LAZ", c_laspy)
check("open3d", c_open3d, critical=False)
check("spconv", c_spconv, critical=False)
check("torch-scatter", c_scatter, critical=False)
check("OpenPCDet", c_pcdet, critical=False)

print("=" * 60)
failed_critical = [r for r in results if not r[0] and r[3]]
failed_optional = [r for r in results if not r[0] and not r[3]]

if failed_critical:
    print(f"{RED}{len(failed_critical)} critical check(s) failed - "
          f"fix before running models.{RESET}")
    for _, name, detail, _ in failed_critical:
        print(f"  - {name}: {detail}")
    sys.exit(1)

if failed_optional:
    print(f"{YELLOW}{len(failed_optional)} optional component(s) missing "
          f"(fine if you are not using them).{RESET}")
    for _, name, _, _ in failed_optional:
        print(f"  - {name}")

print(f"{GREEN}Environment OK.{RESET}\n")
