"""Test-time dependency shims.

This project is GPU-centric (flash-attn + triton + torch). A common unit-test
setup (especially on CPU CI) is to run lightweight tests that don't execute the
GPU kernels but still import core modules.

Shared shim logic lives in tests/_shims.py; this file wires it into pytest.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests._shims import _module_available, install_gpu_shims


def pytest_configure(config):
    # Unit tests should run without requiring a C++ toolchain. Some small layers
    # are decorated with `@torch.compile`, which would otherwise trigger
    # TorchDynamo/Inductor and attempt to build extensions.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    try:  # pragma: no cover
        import torch._dynamo

        torch._dynamo.config.disable = True
    except Exception:
        # If torch isn't installed or internals change, keep tests importable.
        pass

    install_gpu_shims()


def pytest_collection_modifyitems(config, items):
    cuda_visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    explicitly_hidden = cuda_visible_raw is not None and cuda_visible_raw == ""
    cuda_available = False
    if _module_available("torch"):
        import torch

        cuda_available = torch.cuda.is_available()

    should_skip = not cuda_available or explicitly_hidden

    if should_skip:
        skip_gpu = pytest.mark.skip(
            reason="requires a real CUDA GPU (CUDA_VISIBLE_DEVICES explicitly hides devices or no CUDA available)",
        )
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip_gpu)
