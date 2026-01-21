"""Test-time dependency shims.

This project is GPU-centric (flash-attn + triton + torch). A common unit-test
setup (especially on CPU CI) is to run lightweight tests that don't execute the
GPU kernels but still import core modules.

These shims are ONLY used when the real packages aren't installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _module_available(name: str) -> bool:
    # Avoid stubbing real packages that are installed but not yet imported.
    return importlib.util.find_spec(name) is not None


def pytest_configure():
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

    # ---------------------------------------------------------------------
    # flash_attn shim (needed because nanovllm_voxcpm/__init__.py imports llm).
    # ---------------------------------------------------------------------
    if not _module_available("flash_attn"):
        flash_attn = _ensure_module("flash_attn")

        def _unavailable(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("flash_attn is not available in unit tests")

        # Used by nanovllm_voxcpm.layers.attention.
        setattr(flash_attn, "flash_attn_varlen_func", _unavailable)
        setattr(flash_attn, "flash_attn_with_kvcache", _unavailable)
        setattr(flash_attn, "flash_attn_func", _unavailable)

    # ---------------------------------------------------------------------
    # triton shim (so importing attention.py doesn't hard-fail).
    # ---------------------------------------------------------------------
    if not _module_available("triton"):
        triton = _ensure_module("triton")

        def jit(fn=None, **kwargs):  # pragma: no cover
            if fn is None:
                return lambda f: f
            return fn

        setattr(triton, "jit", jit)

    if not _module_available("triton.language"):
        tl = _ensure_module("triton.language")
        # Minimal surface so function annotations resolve.
        setattr(tl, "constexpr", object())

        # Some code paths access `triton.language` as an attribute.
        triton = _ensure_module("triton")
        setattr(triton, "language", tl)

    # ---------------------------------------------------------------------
    # huggingface_hub shim (llm.py imports snapshot_download).
    # ---------------------------------------------------------------------
    if not _module_available("huggingface_hub"):
        hub = _ensure_module("huggingface_hub")

        def snapshot_download(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("huggingface_hub is not available in unit tests")

        setattr(hub, "snapshot_download", snapshot_download)

    # ---------------------------------------------------------------------
    # pydantic shim (used for config typing; avoids import-time failures).
    # ---------------------------------------------------------------------
    if not _module_available("pydantic"):
        pyd = _ensure_module("pydantic")

        class BaseModel:  # pragma: no cover
            pass

        setattr(pyd, "BaseModel", BaseModel)
