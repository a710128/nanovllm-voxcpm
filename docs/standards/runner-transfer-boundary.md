# Inference Transfer and Synchronization Standard

This standard is the actionable version of `docs/adr/0001-runner-owns-host-device-transfers.md`. It must be followed whenever inference code is added or modified.

## 1. Goal

The inference hot path must be divided into four strict phases. The "inference hot path" includes the per-step generation path and the prompt-audio latent-encoding path before generation begins.

1. CPU validation
2. Asynchronous H2D inside the runner
3. GPU-only computation with no synchronization
4. Unified D2H inside the runner

No implementation may change this order.

## 2. Scope

This applies to the inference / generation hot paths in the following directories:

- `nanovllm_voxcpm/engine/`
- `nanovllm_voxcpm/models/voxcpm/`
- Any future `nanovllm_voxcpm/models/*/` directories

## 3. Layering Requirements

### `server`

Allowed:

- Request argument validation
- I/O parsing
- CPU-side data decoding and normalization
- Passing CPU payloads to the engine

Forbidden:

- Any `cuda` / `cpu` / `to(device)` call
- Any operation that reads values from a GPU tensor

### `engine` / `scheduler`

Allowed:

- Scheduling
- KV planning
- `Sequence` state advancement
- Pure CPU post-processing

Forbidden:

- Creating GPU tensors
- Pulling results back from the GPU
- Relying on GPU intermediate state for validation or stop decisions

### `model runner`

Allowed:

- All H2D / D2H transfers
- Attention-context preparation
- GPU forward / graph replay / decode
- Returning final results to the CPU

Constraints:

- H2D must fully prepare inputs before GPU computation begins
- GPU computation must not trigger synchronization
- D2H must occur only after GPU computation finishes
- Pinned memory and `non_blocking=True` are strongly recommended, but they are not a separate goal that replaces the boundary rule itself

### `model` / `layers`

Allowed:

- Pure device-side computation

Forbidden:

- Direct access to Python scalar values derived from GPU results
- `.cpu()` / `.numpy()` / `.tolist()` / `.item()`
- Implicit or explicit Host/Device transfers

## 4. Forbidden Patterns

The following patterns are forbidden everywhere outside the runner:

```python
tensor.cuda()
tensor.to("cuda")
torch.tensor(data, device="cuda")
torch.as_tensor(data, device="cuda")
tensor.cpu()
tensor.numpy()
tensor.tolist()
tensor.item()
torch.cuda.synchronize()
torch.cuda.current_stream().synchronize()
```

Even inside the runner, the following patterns may appear only in Phase D:

```python
tensor.cpu()
tensor.numpy()
tensor.tolist()
tensor.item()
```

Multiple consecutive D2H reads are allowed, but they must happen only after GPU computation finishes and must not overlap with GPU computation.

## 5. Recommended Patterns

### Phase A: CPU Validation

```python
if max_generate_length < 1:
    raise ValueError("max_generate_length must be >= 1")

payload = np.asarray(data, dtype=np.float32)
```

### Phase B: Asynchronous H2D Inside the Runner

```python
gpu_inputs = torch.from_numpy(cpu_array).cuda(non_blocking=True)
gpu_scalar = torch.tensor(values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
```

### Phase C: GPU-Only Computation

```python
outputs = self.run_model(inputs, is_prefill)
latents = outputs["latents"]
decoded = self.vae.decode(latents.permute(0, 2, 1))
```

### Phase D: Unified D2H Inside the Runner

```python
latents_cpu = latents.to(torch.float32).cpu().numpy()
stop_flags_cpu = outputs["stop_flag"].cpu().tolist()
```

## 6. PR / Code Review Checklist

- Are all input validations kept on the CPU?
- Does only the runner perform H2D / D2H?
- Is GPU computation completely free of synchronization?
- Does `postprocess_seq()` receive only CPU data?
- Were any new `.cuda()`, `.cpu()`, `.item()`, or `.tolist()` calls added outside the runner?

## 7. Requirements for New Model Integrations

When adding a new `models/<family>/runner.py`, the following requirements must be explicitly satisfied:

- Input payloads are CPU-side data structures.
- `run()` owns the entire cross-device boundary.
- Results returned to the engine are CPU-side data structures.
- Device semantics do not leak into `engine.py`, `server.py`, or `model.py`.

## 8. Existing Code Baseline

Typical locations that already follow the standard:

- `nanovllm_voxcpm/engine/model_runner.py:292`
- `nanovllm_voxcpm/engine/model_runner.py:297`
- `nanovllm_voxcpm/models/voxcpm/runner.py:102`

Typical location that still needs cleanup:

- `nanovllm_voxcpm/models/voxcpm/server.py:108`
