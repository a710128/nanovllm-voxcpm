# LoRA inference is kernel-launch bound on `openbmb/VoxCPM2`

## TL;DR

After the runtime overhead cleanup landed in `71a84bb`, the residual
LoRA-vs-no-LoRA gap is **dominated by Triton kernel launch overhead** baked
into the CUDA graph at decode time, not by Python-side bookkeeping. Closing it
further requires kernel work, not framework work.

| concurrency | TTFB p50 (s) no-lora → with-lora | RTF/req no-lora → with-lora | with-lora gap |
| --- | --- | --- | --- |
| 1 | 0.212 → 0.372 | 0.117 → 0.150 | **+0.16 s, +29%** |
| 2 | 0.206 → 0.379 | 0.126 → 0.159 | **+0.17 s, +26%** |
| 4 | 0.207 → 0.374 | 0.140 → 0.177 | **+0.17 s, +26%** |
| 8 | 0.202 → 0.390 | 0.173 → 0.211 | **+0.19 s, +22%** |

Reproduction (single GPU, `RTX 4090`, device 4):

```bash
HF_ENDPOINT=https://hf-mirror.com uv run python benchmark/bench_inference.py \
  --model openbmb/VoxCPM2 --devices 4 --concurrency <C> --warmup 1 --iters 3 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000 \
  --max-loras 1 --max-lora-rank 8 --lora-path models/lora_10pct_ref/latest \
  --json-out benchmark/optimized/voxcpM2_with_lora_c<C>.json
```

Raw measurement files are committed under `benchmark/optimized/`.

## Why the gap is now kernel-bound

`openbmb/VoxCPM2` runs **~256 LoRA `shrink`+`expand` Triton kernel launches per
decode step**:

```
base_lm:     28 layers × {qkv, o, gate_up, down}                      = 112
residual_lm:  8 layers × {qkv, o, gate_up, down}                      =  32
feat_decoder.estimator.decoder: 12 layers × {qkv, o, gate_up, down}  ×
                                inference_timesteps                    = ~120
projection LoRALinears (enc_to_lm_proj, fusion_concat_proj, ...)       =   4
                                                                       ───
                                                                       ~256–270
```

The decode CUDA graph is captured once at startup with **every** LoRA layer
present. At replay time those launches are pinned in the graph regardless of
whether the active adapter actually populates that module. The
`models/lora_10pct_ref/latest` checkpoint, for example, only touches
`q_proj/k_proj/v_proj/o_proj`; `gate_up_proj` and `down_proj` adapters run with
all-zero weights every step purely to satisfy graph topology. That is wasted
shrink+expand work on every layer of every step.

Extra evidence the gap is launch-bound, not arithmetic:

- The TTFB regression is a near-constant ~0.17 s additive cost across
  c = 1..8. A purely-arithmetic regression would scale with batch.
- The RTF gap stays ~22–29% across batch sizes (samples/s scales sub-linearly
  in lockstep with no-LoRA samples/s).
- `_resolve_token_slots`'s host-side empty-slot shortcut (skip `add_lora` when
  every active slot has `effective_lora_rank == 0` for this module) **already
  fires** in eager paths, so prefill barely benefits — the regression lives in
  the captured decode graph where the shortcut is intentionally disabled to
  keep replay correct.

## Where the headroom is

In rough order of expected impact ↔ implementation cost.

### 1. BGMV-style fused single-token shrink+expand (high impact)

For decode (`M = 1` token per step, common in TTS workloads), the small-`M`
shrink kernel already special-cases the layout but still launches one
shrink + one expand per layer. A Punica-style **BGMV** kernel that fuses
`y += (x @ A_slot) @ B_slot * scaling` into a single launch per layer would
roughly halve the launch count. Combined with `effective_rank` early-exit
inside the kernel (see #3) this is the most direct path to recovering the
22–29% RTF gap.

References to mirror:
- `vllm/lora/ops/triton_ops/v1/lora_kernel_v1.py` (their fused decode path)
- `punica` original BGMV kernel

Touch points:
- New kernel under `nanovllm_voxcpm/lora_ops/triton_ops/`, parallel to
  `lora_shrink_op.py` / `lora_expand_op.py`.
- Branch in `_VendoredTritonPunicaBackend.add_lora` (`nanovllm_voxcpm/lora.py`)
  selecting the fused path when `x.size(0) <= some_threshold` and slices
  share `(rank, hidden_out)`.

### 2. Cross-layer batched LoRA launch (high impact, harder)

All LoRA layers in a forward pass see the same `LoRAMetadata` and the same
`x` shape. Today each layer issues its own kernel launches sequentially. A
"super-LoRA" entry point that consumes a list of (A, B, y_offset) per layer
and fans them out as one grid would amortize launch latency across the whole
model.

This requires:
- Stable per-step weight pointer arrays (we already cache these via
  `_get_lora_a_ptr` / `_get_lora_b_ptr`).
- A scheduling pass at runner level that gathers the per-layer plan into one
  contiguous descriptor tensor before invoking the model's forward.
- Either deep model-graph cooperation, or a `nn.Module` wrapper that defers
  the actual LoRA add until a sync point at the end of each transformer
  block.

This is invasive — likely a separate PR — but it is the only known path to
make LoRA decode latency converge to base-model decode latency.

### 3. Per-slot `effective_rank` early-exit inside the Triton kernels (medium impact)

Add an early-return in `_lora_shrink_kernel` and `_lora_expand_kernel`:

```python
eff_rank = tl.load(effective_lora_rank_ptr + lora_id)
if eff_rank == 0:
    return
```

Each LoRA `nn.Module` already maintains a per-slot
`effective_lora_rank` int32 tensor on device (`_LoRALayerBase`). Wiring it
through the kernel signature would let the captured decode graph **skip
shrink+expand for modules the active adapter does not cover**, without
recapturing the graph per adapter.

Touch points:
- `nanovllm_voxcpm/lora_ops/triton_ops/lora_shrink_op.py`,
  `nanovllm_voxcpm/lora_ops/triton_ops/lora_expand_op.py`,
  `nanovllm_voxcpm/lora_ops/triton_ops/kernel_utils.py` —
  thread `effective_lora_rank` pointer + early-return.
- `nanovllm_voxcpm/layers/lora.py` — pass the layer's
  `effective_lora_rank` tensor into `add_lora`.
- `nanovllm_voxcpm/lora.py` — extend `LoRAMetadata` and the kernel call
  sites.

The savings here are real but smaller than #1 because we still pay the
launch cost; we only save the inner `tl.dot` work for layers where every
active slot has rank 0.

### 4. Reduce `output_tensor.zero_()` overhead (low impact, easy)

Both shrink paths call `output_tensor.zero_()` as a separate kernel. For
small-`M` decode this is cheap per call but adds up across ~128 launches.
Folding the zero into the first iteration of the kernel (write instead of
add) is a one-line change to each kernel.

### 5. PDL (programmatic dependent launch) (low impact, hardware-gated)

`supports_pdl` currently always returns `False` (`lora_ops/triton_ops/utils.py`).
On Hopper+ this can overlap launch latency with the previous kernel's tail.
The 4090 (sm_89) does not support PDL, so this only helps newer GPUs and is
not relevant to the current benchmark, but worth turning on once we have
H100/H200 numbers.

## Notes on what *not* to chase

- More framework-side micro-optimization in the Python add_lora pipeline.
  Profiling under `nsys`/`torch.profiler` after `71a84bb` shows the residual
  cost is overwhelmingly inside Triton kernels and CUDA launch latency, not
  in `_VendoredTritonPunicaBackend.add_lora` itself.
- `torch.compile` on the LoRA path. The vendored kernels are already Triton;
  the bottleneck is launch count, not kernel codegen.
- More aggressive caching of `_AddLoraPlan` / `_LORA_*_PTR_DICT`. They are
  already lifetime-cached and contribute negligibly to per-step time.

## Acceptance criteria for follow-up work

A LoRA optimization PR should aim for, at the c=1 single-GPU baseline above:

- TTFB p50 with-lora ≤ **1.3×** no-lora (currently 1.75×)
- RTF/req with-lora ≤ **1.10×** no-lora (currently 1.29×)

while keeping all `tests/unit/` tests green and not regressing the no-LoRA
numbers above by more than the run-to-run noise floor (~±2% on this rig).
