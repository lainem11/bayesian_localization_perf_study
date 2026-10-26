# benchmarks/env_light.py
# Environment record that needs only torch and triton, for NVIDIA (CUDA) and AMD (ROCm) machines.
# collect_env() in env_info.py also needs nvidia-smi, JAX, evosax, numpyro and ABL.
# Every probe is optional: anything that fails is recorded as "unavailable", not guessed.
import importlib.metadata
import os
import platform
import shutil
import subprocess
import torch
import triton


def _run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, text=True, cwd=cwd, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as e:                      # missing tool, non-zero exit, not a git repo, ...
        return f"unavailable ({type(e).__name__})"


def _pkg(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_env_light():
    d = {}
    d["GIT branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    d["GIT commit"] = _run(["git", "rev-parse", "HEAD"])
    root = _run(["git", "rev-parse", "--show-toplevel"])
    status = _run(["git", "status", "--porcelain", "--", ".", ":!benchmarks/results"], cwd=root) \
        if not root.startswith("unavailable") else root
    d["GIT dirty"] = status if status.startswith("unavailable") else bool(status.strip())

    d["PyTorch version"] = torch.__version__
    d["torch.version.cuda"] = torch.version.cuda          # None on ROCm builds
    d["torch.version.hip"] = getattr(torch.version, "hip", None)   # None on CUDA builds
    d["GPU available"] = torch.cuda.is_available()        # torch.cuda is also the API for HIP devices
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        d["GPU name"] = props.name
        d["GPU memory"] = f"{props.total_memory / 1024**3:.0f} GiB"
        d["multiprocessors (SMs on NVIDIA, CUs on AMD)"] = props.multi_processor_count
        for attr in ["gcnArchName", "warp_size", "L2_cache_size", "major", "minor"]:
            d[f"props.{attr}"] = getattr(props, attr, "not reported")
        d["GPU count"] = torch.cuda.device_count()
    d["Triton version"] = triton.__version__
    d["pip triton / pytorch-triton-rocm"] = f"{_pkg('triton')} / {_pkg('pytorch-triton-rocm')}"

    if shutil.which("nvidia-smi"):
        d["nvidia-smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if d.get("torch.version.hip"):
        try:
            with open("/opt/rocm/.info/version") as f:
                d["/opt/rocm/.info/version"] = f.read().strip()
        except OSError as e:
            d["/opt/rocm/.info/version"] = f"unavailable ({type(e).__name__})"
        d["amd-smi version"] = _run(["amd-smi", "version"]) if shutil.which("amd-smi") else "amd-smi not found"
    d["HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES"] = f"{os.getenv('HIP_VISIBLE_DEVICES')} / {os.getenv('CUDA_VISIBLE_DEVICES')}"
    d["Python version"] = platform.python_version()
    d["OS"] = platform.platform()
    d["CPU"] = platform.processor() or "not reported"
    return d


if __name__ == "__main__":
    for key, value in collect_env_light().items():
        print(f"{key}: {value}")
