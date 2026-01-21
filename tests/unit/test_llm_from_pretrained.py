import json
import sys
import types

import pytest


def test_from_pretrained_uses_local_path_and_dispatches(monkeypatch, tmp_path):
    # Stub flash_attn to bypass the import guard in nanovllm_voxcpm.llm.
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))

    # Stub huggingface_hub; snapshot_download must not be called for local paths.
    hub = types.ModuleType("huggingface_hub")

    def _snapshot_download(*args, **kwargs):  # pragma: no cover
        raise AssertionError("snapshot_download should not be called for local model paths")

    hub.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    # Stub the VoxCPM server pool classes.
    server_mod = types.ModuleType("nanovllm_voxcpm.models.voxcpm.server")

    class SyncVoxCPMServerPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AsyncVoxCPMServerPool:  # pragma: no cover
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    server_mod.SyncVoxCPMServerPool = SyncVoxCPMServerPool
    server_mod.AsyncVoxCPMServerPool = AsyncVoxCPMServerPool
    monkeypatch.setitem(sys.modules, "nanovllm_voxcpm.models.voxcpm.server", server_mod)

    # The llm module depends on pydantic via LoRAConfig.
    pytest.importorskip("pydantic")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm"}), encoding="utf-8")

    from nanovllm_voxcpm.llm import VoxCPM

    obj = VoxCPM.from_pretrained(model=str(model_dir))
    assert isinstance(obj, SyncVoxCPMServerPool)
    assert obj.kwargs["model_path"] == str(model_dir)
    assert obj.kwargs["devices"] == [0]
