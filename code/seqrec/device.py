"""Device resolution (copied from code/ml-1m/device.py; keep the two in step).

The original code hard-codes .cuda() everywhere, which makes the repo
unrunnable on a CPU box and impossible to smoke-test before paying for a GPU.
Everything now routes through get_device().

Resolution order:
  1. DECRS_DEVICE env var ('cpu' / 'cuda' / 'cuda:0') -- explicit override.
  2. CUDA if torch.cuda.is_available().
  3. CPU.

The lookup is lazy and cached: main.py sets CUDA_VISIBLE_DEVICES from --gpu
before the first call, and resolving eagerly at import time would initialise
CUDA before that assignment takes effect.
"""
import os
import torch

_device = None


def get_device():
    global _device
    if _device is None:
        override = os.environ.get("DECRS_DEVICE", "").strip()
        if override:
            _device = torch.device(override)
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def describe():
    dev = get_device()
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return "device: {} ({}, {:.1f} GiB)".format(dev, name, total)
    return "device: {} (no CUDA visible -- training will be slow)".format(dev)
