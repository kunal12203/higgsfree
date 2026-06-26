"""Shared GPU environment setup for all subprocess pipeline scripts."""
import os

VENV_AVATAR = os.environ.get("VENV_AVATAR", "/home/ubuntu/venv-avatar")
VENV_3D = os.environ.get("VENV_3D", "/home/ubuntu/venv-3d")
VENV_CHATTERBOX = os.environ.get("VENV_CHATTERBOX", "/home/ubuntu/venv-chatterbox")

# CUDA 12 shared libs — needed for onnxruntime-gpu CUDAExecutionProvider
_NVIDIA_LIBS = [
    f"{VENV_AVATAR}/lib/python3.10/site-packages/nvidia/cublas/lib",
    f"{VENV_AVATAR}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    f"{VENV_AVATAR}/lib/python3.10/site-packages/nvidia/cudnn/lib",
    f"{VENV_3D}/lib/python3.10/site-packages/nvidia/cublas/lib",
    f"{VENV_3D}/lib/python3.10/site-packages/nvidia/cudnn/lib",
    f"{VENV_CHATTERBOX}/lib/python3.10/site-packages/nvidia/cublas/lib",
    f"{VENV_CHATTERBOX}/lib/python3.10/site-packages/nvidia/cudnn/lib",
    "/usr/local/cuda/lib64",
    "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib",
]


def gpu_env(base: dict = None) -> dict:
    """Return env dict with LD_LIBRARY_PATH set for GPU onnxruntime + torch."""
    env = (base or os.environ).copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    extras = ":".join(p for p in _NVIDIA_LIBS if os.path.isdir(p))
    env["LD_LIBRARY_PATH"] = extras + (":" + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env
