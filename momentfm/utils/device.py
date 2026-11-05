"""Device selection shared by Linux and Apple Silicon inference tools."""

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU; explicit requests never silently fall back."""
    if name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {name}")
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA is unavailable; check the PyTorch backend and NVIDIA driver.")
    if name == "mps" and not mps_available:
        raise RuntimeError("MPS is unavailable in this process; use --device cpu.")
    return torch.device(name)
