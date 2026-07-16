"""Shared GPU/optional-dep shims for unit and deployment tests.

These are installed in sys.modules ONLY when the real packages aren't available.
Import this module in conftest.py via pytest_configure (or at module level) to
set up shims before any nanovllm_voxcpm code is imported.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _module_available(name: str) -> bool:
    """Return True only if the module can be imported.

    `find_spec()` being non-None just means the import machinery can *locate* the
    module; GPU-centric packages can still fail at import time (missing CUDA,
    incompatible wheels, etc.). For unit tests we treat those as unavailable and
    install a shim instead.
    """

    # Fast negative path and avoids importing heavy modules when absent.
    try:
        if importlib.util.find_spec(name) is None:
            return False
    except (ModuleNotFoundError, AttributeError, ValueError):
        # Some environments may provide partial stubs (e.g. a non-package module
        # named "triton") where importlib's module resolution can error.
        return False

    try:
        importlib.import_module(name)
        return True
    except Exception:
        # Failed imports can leave partial entries in sys.modules; purge the
        # module so our shim can be installed cleanly.
        for mod_name in list(sys.modules.keys()):
            if mod_name == name or mod_name.startswith(name + "."):
                sys.modules.pop(mod_name, None)
        return False


def install_gpu_shims() -> None:
    """Install all standard GPU shims. Call from pytest_configure() or at module level."""

    # -------------------------------------------------------------------------
    # flash_attn shim (needed because nanovllm_voxcpm/__init__.py imports llm).
    # -------------------------------------------------------------------------
    if not _module_available("flash_attn"):
        flash_attn = _ensure_module("flash_attn")

        def _unavailable(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("flash_attn is not available in tests")

        setattr(flash_attn, "flash_attn_varlen_func", _unavailable)
        setattr(flash_attn, "flash_attn_with_kvcache", _unavailable)
        setattr(flash_attn, "flash_attn_func", _unavailable)

    # -------------------------------------------------------------------------
    # triton shim (so importing attention.py doesn't hard-fail).
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # huggingface_hub shim (llm.py imports snapshot_download).
    # -------------------------------------------------------------------------
    if not _module_available("huggingface_hub"):
        hub = _ensure_module("huggingface_hub")

        def snapshot_download(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("huggingface_hub is not available in tests")

        setattr(hub, "snapshot_download", snapshot_download)

    # -------------------------------------------------------------------------
    # pydantic shim (used for config typing; avoids import-time failures).
    # -------------------------------------------------------------------------
    if not _module_available("pydantic"):
        pyd = _ensure_module("pydantic")

        class BaseModel:  # pragma: no cover
            pass

        setattr(pyd, "BaseModel", BaseModel)

    # -------------------------------------------------------------------------
    # transformers shim (needed by voxcpm2 engine and utils imports).
    #
    # Install a shim that covers ALL symbols used across the test suite so
    # that no individual test file needs to inject its own partial stub.
    # A partial stub registered by one test file would lack attributes like
    # LlamaTokenizerFast and break subsequent tests that import the engine.
    # -------------------------------------------------------------------------
    if not _module_available("transformers"):
        t = _ensure_module("transformers")

        class _FakePreTrainedTokenizer:  # pragma: no cover
            pass

        class _FakeLlamaTokenizerFast:  # pragma: no cover
            pass

        class _FakeAutoTokenizer:  # pragma: no cover
            pass

        setattr(t, "PreTrainedTokenizer", _FakePreTrainedTokenizer)
        setattr(t, "LlamaTokenizerFast", _FakeLlamaTokenizerFast)
        setattr(t, "AutoTokenizer", _FakeAutoTokenizer)
