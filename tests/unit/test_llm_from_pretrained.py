import importlib
import asyncio
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


def test_from_pretrained_downloads_remote_model_and_uses_async_pool(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))

    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    (downloaded_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}), encoding="utf-8")

    hub = types.ModuleType("huggingface_hub")

    def _snapshot_download(*, repo_id):
        assert repo_id == "org/model"
        return str(downloaded_dir)

    setattr(cast(Any, hub), "snapshot_download", _snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    server_mod = types.ModuleType("nanovllm_voxcpm.models.voxcpm2.server")

    class SyncServerPool:  # pragma: no cover
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AsyncServerPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(server_mod, "SyncVoxCPM2ServerPool", SyncServerPool)
    setattr(server_mod, "AsyncVoxCPM2ServerPool", AsyncServerPool)
    monkeypatch.setitem(sys.modules, "nanovllm_voxcpm.models.voxcpm2.server", server_mod)

    pytest.importorskip("pydantic")

    sys.modules.pop("nanovllm_voxcpm.llm", None)
    llm = importlib.import_module("nanovllm_voxcpm.llm")

    async def run():
        return llm.VoxCPM.from_pretrained(model="org/model", devices=[3], extra_flag=True)

    obj = asyncio.run(run())
    assert isinstance(obj, AsyncServerPool)
    assert obj.kwargs["model_path"] == str(downloaded_dir)
    assert obj.kwargs["devices"] == [3]
    assert obj.kwargs["extra_flag"] is True


def test_from_pretrained_rejects_missing_tilde_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    hub = types.ModuleType("huggingface_hub")
    setattr(cast(Any, hub), "snapshot_download", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    pytest.importorskip("pydantic")

    sys.modules.pop("nanovllm_voxcpm.llm", None)
    llm = importlib.import_module("nanovllm_voxcpm.llm")

    with pytest.raises(ValueError, match="does not exist"):
        llm.VoxCPM.from_pretrained(model="~/missing-model-dir")


def test_from_pretrained_rejects_missing_config_file(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    hub = types.ModuleType("huggingface_hub")
    setattr(cast(Any, hub), "snapshot_download", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    pytest.importorskip("pydantic")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    sys.modules.pop("nanovllm_voxcpm.llm", None)
    llm = importlib.import_module("nanovllm_voxcpm.llm")

    with pytest.raises(FileNotFoundError, match="Config file"):
        llm.VoxCPM.from_pretrained(model=str(model_dir))


def test_from_pretrained_rejects_unsupported_architecture(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    hub = types.ModuleType("huggingface_hub")
    setattr(cast(Any, hub), "snapshot_download", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    pytest.importorskip("pydantic")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "unknown"}), encoding="utf-8")

    sys.modules.pop("nanovllm_voxcpm.llm", None)
    llm = importlib.import_module("nanovllm_voxcpm.llm")

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        llm.VoxCPM.from_pretrained(model=str(model_dir))
