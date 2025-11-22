"""Common helpers for probing CUDA/GPU support."""

import importlib.util
import os
import platform
import shutil
import subprocess

SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None


def gpu_supported() -> bool:
    """Return ``True`` if CUDA/GPU helpers should be enabled."""

    return IS_WINDOWS and CUPY_AVAILABLE


def enforce_nvidia_gpu() -> bool:
    """Force usage of the first NVIDIA GPU via ``CUDA_VISIBLE_DEVICES``."""

    if not gpu_supported():
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        index = output.splitlines()[0].strip()
        if index:
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            return True
    except Exception:
        pass
    return False


__all__ = ["gpu_supported", "enforce_nvidia_gpu", "IS_WINDOWS", "CUPY_AVAILABLE"]
