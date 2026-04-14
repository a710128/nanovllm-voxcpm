# ADR 0001: The Model Runner Exclusively Owns the Host/Device Transfer and Synchronization Boundary

## Status

Accepted

## Date

2026-04-14

## Context

The current inference pipeline is composed of `server -> engine -> scheduler -> model runner -> model -> engine postprocess`.

Based on the current implementation, the main Host/Device boundaries in the inference hot path are located at the following points:

- `nanovllm_voxcpm/models/voxcpm/server.py:108` executes `wav_tensor.cuda()` in the service layer.
- `nanovllm_voxcpm/engine/model_runner.py:292` to `nanovllm_voxcpm/engine/model_runner.py:360` build the attention context and perform Host-to-Device transfers inside the runner.
- `nanovllm_voxcpm/models/voxcpm/runner.py:125` to `nanovllm_voxcpm/models/voxcpm/runner.py:132` asynchronously move the payload from CPU to GPU inside the runner.
- `nanovllm_voxcpm/models/voxcpm/runner.py:151` to `nanovllm_voxcpm/models/voxcpm/runner.py:157`, and `nanovllm_voxcpm/models/voxcpm/runner.py:170`, move results back from GPU to CPU inside the runner.

This means the current code already places most data movement on the runner side, but two issues remain:

1. The Host/Device boundary has not been formally declared as an architectural constraint.
2. The service layer still contains a direct `cuda()` call, which breaks boundary consistency.

At the same time, the inference pipeline is sensitive to both throughput and tail latency. The "inference hot path" here includes two kinds of execution:

- The per-step generation path.
- The latent-encoding path for prompt audio before generation starts.

Any Host/Device transfer or implicit synchronization scattered across `server`, `engine`, `scheduler`, `model`, or `layer` can lead to the following problems:

- It becomes difficult to reason about the execution order and blocking points of a single step.
- It becomes difficult to ensure the stability of CUDA graph or stream-based execution.
- Synchronization can easily be introduced through seemingly harmless calls such as `.cpu()`, `.item()`, `.tolist()`, or `.numpy()`.
- Responsibilities for CPU validation, GPU computation, and result transfer become mixed together, reducing maintainability.

## Decision

Starting with this ADR, the inference hot path must follow the boundary rules below:

1. All Host-to-Device and Device-to-Host transfers may occur only in the `model runner` layer.
2. All cross-boundary reads that may trigger synchronization must also occur only in the `model runner` layer.
3. The inference hot path must follow this strict order:
   - First, complete all validation and normalization on the CPU.
   - Then, let the runner asynchronously move inputs to the GPU.
   - Then, complete all computation on the GPU without triggering any synchronization.
   - Finally, let the runner move results back from the GPU to the CPU.
4. `server`, `engine`, `scheduler`, `model`, and `layers` must not directly perform any Host/Device boundary-crossing operations in the inference hot path.

## Detailed Constraints

### 1. Layer Responsibilities

#### CPU-side outer layer: `server` / `engine` / `scheduler`

May only do the following:

- Request parsing, argument validation, length checks, and type checks.
- Python- or NumPy-level data normalization.
- `Sequence` / `RunnerTask` construction and state advancement.
- Scheduling, KV-block planning, and stop-condition checks.

Must not do the following:

- `tensor.cuda()`
- `tensor.to("cuda")`, `module.to("cuda")`
- `torch.tensor(..., device="cuda")`
- `torch.as_tensor(..., device="cuda")`
- `tensor.cpu()`
- `tensor.numpy()` (when the tensor is still on the GPU)
- `tensor.tolist()`, `tensor.item()`
- `torch.cuda.synchronize()`
- `torch.cuda.current_stream().synchronize()`
- Any implicit synchronization performed to read GPU results

#### GPU boundary layer: `BaseModelRunner` and its subclasses

Are responsible for the following:

- Asynchronously moving already-validated CPU data to the GPU.
- Preparing attention context, KV-cache write locations, and other GPU execution metadata.
- Executing GPU computation such as model forward passes, CUDA graph replay, and VAE decode.
- Returning results to the CPU in a single controlled place after GPU computation finishes.

### 2. Strict Execution Order

Every inference step must follow the four phases below, with no overlap between phases:

#### Phase A: CPU Validation and Normalization

Location: `server`, `engine`, `scheduler`

Requirements:

- Complete validation for shape, dtype, length, value range, and request consistency.
- Complete prompt concatenation, prefix-cache input normalization, and stop-condition upper-bound calculation.
- Produce a complete and self-consistent CPU payload.

After this phase ends, later stages must not read GPU intermediates back to the CPU for any "additional validation."

#### Phase B: Asynchronous H2D Inside the Runner

Location: `BaseModelRunner` / `VoxCPMRunner`

Requirements:

- H2D must be initiated centrally by the runner.
- When the framework and data layout allow it, asynchronous H2D should prefer pinned host memory with `non_blocking=True`.
- Prepare all GPU inputs and execution context needed for the current step in one place.
- Model code, attention layers, and utility helpers must not perform hidden H2D transfers.

#### Phase C: GPU-Only Computation

Location: after the runner invokes the model and before D2H begins

Requirements:

- Only GPU kernels, CUDA graph replay, and tensor transformations on the device are allowed.
- No operation may synchronize control back to the CPU.
- Reading GPU scalars into Python is not allowed.
- `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, and explicit stream synchronization are not allowed.

#### Phase D: Unified D2H Inside the Runner

Location: `BaseModelRunner` / `VoxCPMRunner`

Requirements:

- Move final results back to the CPU only after all GPU computation has completed.
- Multiple consecutive D2H reads are allowed, but they must all remain within Phase D and must not overlap with Phase C.
- Objects produced after D2H must be Python or NumPy data that the engine can consume directly.
- The engine's `postprocess_seq()` must not access GPU tensors.

### 3. Allowed Exceptions

The following exceptions are not part of the "inference hot path," but they must still remain at the runner layer:

- Distributed synchronization and CUDA graph capture during initialization.
- Resource cleanup and synchronization during shutdown.

Even as exceptions, they must not be pushed down into `server`, `engine`, `scheduler`, `model`, or `layers`.

## Direct Consequences

### Benefits

- A clearer Host/Device boundary makes inference performance analysis easier.
- Hidden synchronization points become easier to detect.
- The risk of device semantics leaking into the model and engine layers is reduced.
- It becomes easier to add profiling, stream management, and transfer auditing to the runner later.

### Costs

- The runner takes on more responsibility and must explicitly handle input marshaling and output unmarshaling.
- Some legacy implementations must be moved back from `server` or `engine` into the runner.
- Code review must explicitly check whether this boundary has been crossed.

## Repository Rules

### Required

- `Sequence`, `RunnerTask`, and `custom_payload` must be CPU-serializable data before entering the runner.
- Inputs to `postprocess_seq()` must already be CPU data.
- Creation, copying, and return of all GPU tensors must be handled centrally by the runner.

### Known Places That Still Need Alignment

- `nanovllm_voxcpm/models/voxcpm/server.py:108` to `nanovllm_voxcpm/models/voxcpm/server.py:117`
  The current `encode_latents()` implementation sends waveform data directly to the GPU in the service layer. This logic should be migrated so that the service layer only performs audio decoding and validation, while the runner exclusively owns H2D transfer and encoding execution.

### Recommended Implementation Pattern

- CPU layer: `bytes` / `list` / `np.ndarray` / scalars.
- Runner entry: centrally perform `torch.from_numpy(...)`, pinned-memory handling, and `.cuda(non_blocking=True)`.
- Runner exit: centrally perform `.cpu()` followed by NumPy/Python unpacking.
- Engine layer: handle only CPU results and keep no references to GPU tensors.

## Review Checklist

Any of the following patterns is considered a violation of this ADR by default, unless the code is in the runner and follows the four-phase sequence:

- `.cuda(`
- `.to("cuda")`
- `.to(device=`
- `device="cuda"`
- `.cpu()`
- `.numpy()`
- `.tolist()`
- `.item()`
- `torch.cuda.synchronize()`
- `stream.synchronize()`

## Alternatives Considered

### Option A: Allow the engine to handle a small number of transfers

Rejected. This would make it impossible to reason about whether the scheduling layer blocks, and it would continue expanding the spread of device semantics.

### Option B: Allow the model to return some results by itself

Rejected. This would break the design in which the runner is the sole device boundary and would bury synchronization points deep inside lower-level modules.

## Follow-up Actions

1. Move GPU transfer and execution in `server.encode_latents()` into the runner.
2. Add Host/Device boundary checks to code review.
3. Require in new model integration docs that any cross-device operation may appear only in the runner.
