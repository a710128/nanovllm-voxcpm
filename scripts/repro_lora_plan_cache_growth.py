#!/usr/bin/env python3
"""Check LoRA add-plan cache stability in QKV/Merged LoRA layers.

This script does not need real VoxCPM weights. It builds tiny LoRA-enabled
layers, repeatedly runs forward passes, and prints the vendored LoRA backend's
``_add_lora_plan_cache`` size.

Expected fixed behavior: each case warms one or a few stable plan entries, then
the cache size remains bounded even though QKV/Merged forwards create fresh
temporary LoRA views on every step.

Run:
    uv run python scripts/repro_lora_plan_cache_growth.py --iters 5000
"""

from __future__ import annotations

import argparse
import gc

import torch

from nanovllm_voxcpm.layers.lora import LoRALinear, LoRAMergedColumnParallelLinear, LoRAQKVParallelLinear
from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend, get_backend, set_backend_for_testing
from nanovllm_voxcpm.utils.context import LoRAContext, reset_all_contexts, set_lora_context


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to exercise the vendored Triton LoRA backend")


def _set_single_slot_context(num_tokens: int) -> None:
    set_lora_context(
        LoRAContext(
            token_to_slot=torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
            token_indices_sorted_by_slot=torch.arange(num_tokens, dtype=torch.int32, device="cuda"),
            active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            num_tokens_per_slot=torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
            slot_start_offsets=torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
            no_lora_flag=False,
            num_active_loras=1,
        )
    )


def _plan_cache_size() -> int:
    backend = get_backend()
    return len(getattr(backend, "_add_lora_plan_cache", {}))


def _report(label: str, step: int) -> None:
    torch.cuda.synchronize()
    print(
        f"{label:8s} step={step:6d} "
        f"plan_cache={_plan_cache_size():6d} "
        f"allocated={torch.cuda.memory_allocated():10d} "
        f"reserved={torch.cuda.memory_reserved():10d}",
        flush=True,
    )


@torch.inference_mode()
def _run_direct_qkv_views(
    layer: LoRAQKVParallelLinear,
    hidden_size: int,
    iters: int,
    report_every: int,
) -> tuple[int, int]:
    """Stress the cache key with fresh retained QKV LoRA views.

    QKV/Merged layer forwards create temporary ``self.lora_A[...]`` views. The
    Older cache keys used ``id(view)`` and grew by one entry per iteration here.
    The fixed shape/layout key should keep the cache bounded.
    """

    retained_views: list[tuple[torch.Tensor, ...]] = []
    before = _plan_cache_size()
    for step in range(1, iters + 1):
        num_tokens = 1 + (step % 8)
        _set_single_slot_context(num_tokens)
        x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
        y_packed = torch.randn(num_tokens, layer.q_size + 2 * layer.kv_size, device="cuda", dtype=torch.float16)
        y_slices = list(y_packed.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1))
        lora_a_slices = tuple(layer.lora_A[layer.target_to_index[target]] for target in layer.lora_targets)
        retained_views.append(lora_a_slices)
        lora_b_slices = [getattr(layer, f"lora_B_{target}") for target in layer.lora_targets]
        metadata = layer._runtime_metadata()
        get_backend().add_lora(
            y_slices,
            x,
            list(lora_a_slices),
            lora_b_slices,
            indices=metadata.token_to_slot,
            metadata=metadata,
            scaling=1.0,
            y_packed=y_packed,
        )
        if step == 1 or step % report_every == 0 or step == iters:
            _report("direct", step)
    return before, _plan_cache_size()


@torch.inference_mode()
def _run_layer(label: str, layer: torch.nn.Module, hidden_size: int, iters: int, report_every: int) -> tuple[int, int]:
    before = _plan_cache_size()
    for step in range(1, iters + 1):
        # Vary M to mimic changing decode batch sizes. M < 32 also exercises
        # the small-M Triton path used by decode.
        num_tokens = 1 + (step % 8)
        _set_single_slot_context(num_tokens)
        x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
        _ = layer(x)
        if step == 1 or step % report_every == 0 or step == iters:
            gc.collect()
            _report(label, step)
    return before, _plan_cache_size()


def _make_qkv_layer() -> LoRAQKVParallelLinear:
    layer = LoRAQKVParallelLinear(
        hidden_size=64,
        head_size=8,
        total_num_heads=4,
        total_num_kv_heads=2,
        max_loras=1,
        max_lora_rank=8,
    ).cuda().half()
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.02)
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.randn(3, 8, 64, device="cuda", dtype=torch.float16),
            lora_b=[
                torch.randn(layer.q_size, 8, device="cuda", dtype=torch.float16),
                torch.randn(layer.kv_size, 8, device="cuda", dtype=torch.float16),
                torch.randn(layer.kv_size, 8, device="cuda", dtype=torch.float16),
            ],
            effective_rank=8,
            scaling=1.0,
        )
    return layer


def _make_merged_layer() -> LoRAMergedColumnParallelLinear:
    layer = LoRAMergedColumnParallelLinear(
        input_size=64,
        output_sizes=[64, 64],
        max_loras=1,
        max_lora_rank=8,
    ).cuda().half()
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.02)
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.randn(2, 8, 64, device="cuda", dtype=torch.float16),
            lora_b=[
                torch.randn(64, 8, device="cuda", dtype=torch.float16),
                torch.randn(64, 8, device="cuda", dtype=torch.float16),
            ],
            effective_rank=8,
            scaling=1.0,
        )
    return layer


def _make_linear_layer() -> LoRALinear:
    layer = LoRALinear(64, 64, max_loras=1, max_lora_rank=8).cuda().half()
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.02)
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.randn(8, 64, device="cuda", dtype=torch.float16),
            lora_b=torch.randn(64, 8, device="cuda", dtype=torch.float16),
            effective_rank=8,
            scaling=1.0,
        )
    return layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--report-every", type=int, default=1000)
    args = parser.parse_args()

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.report_every <= 0:
        raise ValueError("--report-every must be > 0")

    _require_cuda()
    torch.manual_seed(1234)
    set_backend_for_testing(_VendoredTritonPunicaBackend())

    try:
        qkv_layer = _make_qkv_layer()
        direct_before, direct_after = _run_direct_qkv_views(qkv_layer, 64, args.iters, args.report_every)
        qkv_before, qkv_after = _run_layer("qkv", qkv_layer, 64, args.iters, args.report_every)
        merged_before, merged_after = _run_layer("merged", _make_merged_layer(), 64, args.iters, args.report_every)
        linear_before, linear_after = _run_layer("linear", _make_linear_layer(), 64, args.iters, args.report_every)
    finally:
        reset_all_contexts()
        set_backend_for_testing(None)

    print(
        "summary "
        f"direct_delta={direct_after - direct_before} "
        f"qkv_delta={qkv_after - qkv_before} "
        f"merged_delta={merged_after - merged_before} "
        f"linear_delta={linear_after - linear_before}"
    )
    if direct_after > direct_before + 2:
        raise RuntimeError("Direct QKV view plan cache grew unexpectedly")
    if qkv_after > qkv_before + 2:
        raise RuntimeError("QKV plan cache grew unexpectedly")
    if merged_after > merged_before + 2:
        raise RuntimeError("Merged plan cache grew unexpectedly")
    if linear_after > linear_before + 1:
        raise RuntimeError("Linear control cache grew unexpectedly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
