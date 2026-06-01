from __future__ import annotations

import os

import torch


def get_default_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU, unless KLDMPLUS_DEVICE is set."""
    forced = os.environ.get("KLDMPLUS_DEVICE", "").strip().lower()
    if forced:
        if forced not in {"cuda", "mps", "cpu"}:
            raise ValueError("KLDMPLUS_DEVICE must be one of: cuda, mps, cpu.")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("KLDMPLUS_DEVICE=cuda requested, but CUDA is not available.")
        if forced == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not torch.backends.mps.is_available():
                raise RuntimeError("KLDMPLUS_DEVICE=mps requested, but MPS is not available.")
        return torch.device(forced)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def should_pin_memory(device: torch.device) -> bool:
    """Pinned host memory is only useful for CUDA transfers."""
    return device.type == "cuda"
