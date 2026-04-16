from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload
from nanovllm_voxcpm.models.voxcpm.model import VoxCPMModel


_LORA_WEIGHTS_FILE = "lora_weights.safetensors"
_LORA_CONFIG_FILE = "lora_config.json"
_TARGET_ORDER = {"q": 0, "k": 1, "v": 2}


@dataclass
class _CollectedModule:
    module_name: str
    module_kind: str
    targets: list[str | int] = field(default_factory=list)
    lora_a_by_target: dict[str | int, torch.Tensor] = field(default_factory=dict)
    lora_b_by_target: dict[str | int, torch.Tensor] = field(default_factory=dict)


def load_voxcpm_lora_checkpoint(path: str, *, tp_size: int = 1) -> LoRAModelPayload | list[LoRAModelPayload]:
    checkpoint_dir = Path(path)
    _validate_checkpoint_dir(checkpoint_dir)
    checkpoint_config = _load_checkpoint_config(checkpoint_dir)
    alpha = _resolve_alpha(checkpoint_config)

    collected_modules = _load_collected_modules(checkpoint_dir)
    if not collected_modules:
        raise ValueError("LoRA checkpoint does not contain any supported module weights")

    rank = _resolve_rank(collected_modules, checkpoint_config)
    base_modules = {name: _build_module_payload(module, rank, alpha) for name, module in collected_modules.items()}
    base_payload = LoRAModelPayload(modules=base_modules, rank=rank, alpha=alpha)
    if tp_size <= 1:
        return base_payload
    return [_shard_payload_for_rank(base_payload, rank_idx, tp_size) for rank_idx in range(tp_size)]


def _validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_dir}")
    weights_path = checkpoint_dir / _LORA_WEIGHTS_FILE
    config_path = checkpoint_dir / _LORA_CONFIG_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing LoRA weights file: {weights_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LoRA config file: {config_path}")


def _load_checkpoint_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / _LORA_CONFIG_FILE
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    if not isinstance(raw_config, dict):
        raise ValueError("LoRA config must be a JSON object")
    lora_config = raw_config.get("lora_config")
    if not isinstance(lora_config, dict):
        raise ValueError("lora_config.json must contain a 'lora_config' object")
    return lora_config


def _resolve_alpha(checkpoint_config: dict) -> float:
    alpha = checkpoint_config.get("alpha", checkpoint_config.get("lora_alpha"))
    if alpha is None:
        raise ValueError("LoRA config must define alpha or lora_alpha")
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be > 0, got {alpha}")
    return alpha


def _resolve_rank(collected_modules: dict[str, _CollectedModule], checkpoint_config: dict) -> int:
    rank_from_config = checkpoint_config.get("r", checkpoint_config.get("rank"))
    inferred_ranks = {
        tensor.size(0) for module in collected_modules.values() for tensor in module.lora_a_by_target.values()
    }
    if not inferred_ranks:
        raise ValueError("LoRA checkpoint does not contain any LoRA A tensors")
    if len(inferred_ranks) != 1:
        raise ValueError(f"LoRA checkpoint contains inconsistent ranks: {sorted(inferred_ranks)}")
    inferred_rank = inferred_ranks.pop()
    if rank_from_config is None:
        return inferred_rank
    rank = int(rank_from_config)
    if rank <= 0:
        raise ValueError(f"LoRA rank must be > 0, got {rank}")
    if rank != inferred_rank:
        raise ValueError(f"LoRA config rank {rank} does not match tensor rank {inferred_rank}")
    return rank


def _load_collected_modules(checkpoint_dir: Path) -> dict[str, _CollectedModule]:
    collected: dict[str, _CollectedModule] = {}
    weights_path = checkpoint_dir / _LORA_WEIGHTS_FILE
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            parsed = _parse_weight_key(key)
            if parsed is None:
                continue
            module_name, module_kind, target, tensor_kind = parsed
            module = collected.setdefault(
                module_name, _CollectedModule(module_name=module_name, module_kind=module_kind)
            )
            if target not in module.targets:
                module.targets.append(target)
            tensor = f.get_tensor(key).to(dtype=torch.float32)
            if tensor_kind == "a":
                module.lora_a_by_target[target] = tensor
            else:
                module.lora_b_by_target[target] = tensor

    for module in collected.values():
        missing_a = [target for target in module.targets if target not in module.lora_a_by_target]
        missing_b = [target for target in module.targets if target not in module.lora_b_by_target]
        if missing_a or missing_b:
            raise ValueError(
                f"Incomplete LoRA tensors for module '{module.module_name}': missing_a={missing_a}, missing_b={missing_b}"
            )
    return collected


def _parse_weight_key(key: str) -> tuple[str, str, str | int, str] | None:
    if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
        raise ValueError("Unsupported LoRA tensor suffix; expected keys ending in '.lora_A' or '.lora_B'")
    if key.endswith(".lora_A"):
        tensor_kind = "a"
        base_key = key[: -len(".lora_A")]
    elif key.endswith(".lora_B"):
        tensor_kind = "b"
        base_key = key[: -len(".lora_B")]
    else:
        return None

    base_key = _strip_known_prefixes(base_key)
    if "." in base_key:
        parent, leaf = base_key.rsplit(".", 1)
    else:
        parent, leaf = "", base_key
    packed_mapping = VoxCPMModel.packed_modules_mapping.get(leaf)
    if packed_mapping is not None:
        packed_module, target = packed_mapping
        if isinstance(target, str):
            module_kind = "qkv"
        else:
            module_kind = "merged_column"
        module_name = f"{parent}.{packed_module}" if parent else packed_module
        return module_name, module_kind, target, tensor_kind

    if leaf in {"o_proj", "down_proj"}:
        return base_key, "row_parallel", leaf, tensor_kind
    return base_key, "linear", leaf, tensor_kind


def _strip_known_prefixes(module_name: str) -> str:
    prefixes = ("base_model.model.", "model.", "module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if module_name.startswith(prefix):
                module_name = module_name[len(prefix) :]
                changed = True
    return module_name


def _build_module_payload(module: _CollectedModule, rank: int, alpha: float) -> LoRAModulePayload:
    scaling = alpha / rank
    ordered_targets = _ordered_targets(module)
    if module.module_kind in {"qkv", "merged_column"}:
        lora_a = torch.stack([module.lora_a_by_target[target] for target in ordered_targets], dim=0)
        lora_b = [module.lora_b_by_target[target] for target in ordered_targets]
        return LoRAModulePayload(lora_a=lora_a, lora_b=lora_b, effective_rank=rank, scaling=scaling)

    target = ordered_targets[0]
    return LoRAModulePayload(
        lora_a=module.lora_a_by_target[target],
        lora_b=module.lora_b_by_target[target],
        effective_rank=rank,
        scaling=scaling,
    )


def _ordered_targets(module: _CollectedModule) -> list[str | int]:
    if module.module_kind == "qkv":
        return sorted(module.targets, key=lambda target: _TARGET_ORDER[str(target)])
    if module.module_kind == "merged_column":
        return sorted(module.targets, key=int)
    return module.targets


def _shard_payload_for_rank(payload: LoRAModelPayload, rank: int, tp_size: int) -> LoRAModelPayload:
    modules: dict[str, LoRAModulePayload] = {}
    for module_name, module_payload in payload.modules.items():
        modules[module_name] = _shard_module_payload(module_name, module_payload, rank, tp_size)
    return LoRAModelPayload(modules=modules, rank=payload.rank, alpha=payload.alpha)


def _shard_module_payload(
    module_name: str, module_payload: LoRAModulePayload, rank: int, tp_size: int
) -> LoRAModulePayload:
    if module_name.endswith("qkv_proj"):
        assert isinstance(module_payload.lora_b, list)
        sharded_b = [tensor.chunk(tp_size, dim=0)[rank].contiguous() for tensor in module_payload.lora_b]
        return LoRAModulePayload(
            lora_a=module_payload.lora_a.clone(),
            lora_b=sharded_b,
            effective_rank=module_payload.effective_rank,
            scaling=module_payload.scaling,
        )
    if module_name.endswith("gate_up_proj"):
        assert isinstance(module_payload.lora_b, list)
        sharded_b = [tensor.chunk(tp_size, dim=0)[rank].contiguous() for tensor in module_payload.lora_b]
        return LoRAModulePayload(
            lora_a=module_payload.lora_a.clone(),
            lora_b=sharded_b,
            effective_rank=module_payload.effective_rank,
            scaling=module_payload.scaling,
        )
    if module_name.endswith("o_proj") or module_name.endswith("down_proj"):
        assert isinstance(module_payload.lora_b, torch.Tensor)
        sharded_a = module_payload.lora_a.chunk(tp_size, dim=1)[rank].contiguous()
        return LoRAModulePayload(
            lora_a=sharded_a,
            lora_b=module_payload.lora_b.clone(),
            effective_rank=module_payload.effective_rank,
            scaling=module_payload.scaling,
        )
    return LoRAModulePayload(
        lora_a=module_payload.lora_a.clone(),
        lora_b=(
            [tensor.clone() for tensor in module_payload.lora_b]
            if isinstance(module_payload.lora_b, list)
            else module_payload.lora_b.clone()
        ),
        effective_rank=module_payload.effective_rank,
        scaling=module_payload.scaling,
    )
