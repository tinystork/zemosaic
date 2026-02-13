def _fmt_cuda_int(v: int) -> str:
    # v typiquement 12010 -> "12.1"
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"

def show_cuda_versions():
    print("=== CUDA versions (best effort) ===")

    # --- CuPy (runtime/driver réels) ---
    try:
        import cupy as cp
        rt = cp.cuda.runtime.runtimeGetVersion()
        dr = cp.cuda.runtime.driverGetVersion()
        print(f"CuPy: {cp.__version__}")
        print(f"CUDA Runtime (via CuPy): {_fmt_cuda_int(rt)} ({rt})")
        print(f"CUDA Driver  (via CuPy): {_fmt_cuda_int(dr)} ({dr})")
    except Exception as e:
        print(f"CuPy: not available ({e})")

    # --- PyTorch (souvent version compilée) ---
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"torch.version.cuda (build): {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch: not available ({e})")

    # --- TensorFlow (souvent version compilée) ---
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        try:
            info = tf.sysconfig.get_build_info()
            print(f"TF build cuda_version: {info.get('cuda_version')}")
            print(f"TF build cudnn_version: {info.get('cudnn_version')}")
        except Exception:
            pass
        print(f"TF GPUs: {tf.config.list_physical_devices('GPU')}")
    except Exception as e:
        print(f"TensorFlow: not available ({e})")

    # --- nvcc (toolkit) ---
    import shutil, subprocess
    if shutil.which("nvcc"):
        try:
            out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
            print("\n[nvcc --version]\n" + out.strip())
        except Exception as e:
            print(f"nvcc: error ({e})")
    else:
        print("nvcc: not found (CUDA Toolkit probably not installed / not in PATH)")

    # --- nvidia-smi (driver) ---
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
            print("\n[nvidia-smi]\n" + out.strip())
        except Exception as e:
            print(f"nvidia-smi: error ({e})")
    else:
        print("nvidia-smi: not found (no NVIDIA driver or not in PATH)")

show_cuda_versions()
