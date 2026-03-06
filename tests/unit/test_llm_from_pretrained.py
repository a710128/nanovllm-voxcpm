import importlib
import json
import sys
import types
from typing import Any, cast

import pytest


@pytest.mark.parametrize(
    ("architecture", "module_name", "sync_class_name"),
    [
        ("voxcpm", "nanovllm_voxcpm.models.voxcpm.server", "SyncVoxCPMServerPool"),
        ("voxcpm2", "nanovllm_voxcpm.models.voxcpm2.server", "SyncVoxCPM2ServerPool"),
    ],
)
def test_from_pretrained_uses_local_path_and_dispatches(
    monkeypatch, tmp_path, architecture, module_name, sync_class_name
):
    # Stub flash_attn to bypass the import guard in nanovllm_voxcpm.llm.
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))

    # Stub huggingface_hub; snapshot_download must not be called for local paths.
    hub = types.ModuleType("huggingface_hub")

    def _snapshot_download(*args, **kwargs):  # pragma: no cover
        raise AssertionError("snapshot_download should not be called for local model paths")

    setattr(cast(Any, hub), "snapshot_download", _snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    # Stub the model server pool classes.
    server_mod = types.ModuleType(module_name)

    class SyncServerPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AsyncServerPool:  # pragma: no cover
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(server_mod, sync_class_name, SyncServerPool)
    setattr(server_mod, sync_class_name.replace("Sync", "Async", 1), AsyncServerPool)
    monkeypatch.setitem(sys.modules, module_name, server_mod)

    # The llm module depends on pydantic via LoRAConfig.
    pytest.importorskip("pydantic")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": architecture}), encoding="utf-8")

    sys.modules.pop("nanovllm_voxcpm.llm", None)
    llm = importlib.import_module("nanovllm_voxcpm.llm")

    obj = llm.VoxCPM.from_pretrained(model=str(model_dir))
    assert isinstance(obj, SyncServerPool)
    assert obj.kwargs["model_path"] == str(model_dir)
    assert obj.kwargs["devices"] == [0]
