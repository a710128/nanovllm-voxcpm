# ADR 0002: LoRA Lifecycle and Residency Split Between Engine Control State and Runner Runtime State

## Status

Accepted

## Date

2026-04-15

## Context

The repository already supports request-scoped LoRA registration and execution across `ServerPool -> Server -> Engine -> Runner -> LoRA layers`, but the design needs to be made explicit because the LoRA implementation spans both CPU-side lifecycle management and GPU-side runtime residency.

The current implementation establishes the following facts:

- `nanovllm_voxcpm/models/voxcpm/server.py:574` and `nanovllm_voxcpm/models/voxcpm2/server.py:496` make `ServerPool` the public LoRA truth source for `register_lora`, `unregister_lora`, `list_loras`, and `generate(..., lora_name=...)`.
- `nanovllm_voxcpm/engine/llm_engine.py:120` keeps an engine-side `LoRAManager` for `name -> adapter_id`, request lifecycle callbacks, draining semantics, and scheduler admission checks.
- `nanovllm_voxcpm/engine/model_runner.py:170` keeps a runner-side `LoRARuntime` for rank-local payload registration, GPU slot residency, slot eviction, batch planning, and LoRA execution metadata.
- `nanovllm_voxcpm/engine/lora_manager.py:229` and `nanovllm_voxcpm/engine/lora_manager.py:245` now explicitly separate these two responsibilities into `LoRAManager` and `LoRARuntime`.
- `nanovllm_voxcpm/utils/context.py:18` and `nanovllm_voxcpm/layers/lora.py:146` define a dedicated LoRA runtime context that LoRA-capable layers consume independently from attention context.
- `nanovllm_voxcpm/engine/model_runner.py:555` to `nanovllm_voxcpm/engine/model_runner.py:766` show that LoRA execution must remain compatible with CUDA graph capture and replay.

At the same time, the design must be unambiguous about three boundaries:

1. Public registration state vs local execution state.
2. CPU resident adapter state vs GPU resident slot state.
3. Stable request binding identity (`adapter_id`) vs temporary execution placement (`slot_id`).

Without a written decision, future changes can easily re-couple control-plane and runtime-plane logic, or leak GPU residency assumptions into scheduler and API layers.

## Decision

Starting with this ADR, LoRA support in this repository follows the rules below:

1. `ServerPool` owns the public LoRA name registry and request admission visibility.
2. `Engine` owns LoRA control-plane lifecycle state per server process through `LoRAManager`.
3. `Runner` owns LoRA runtime residency and execution metadata per rank through `LoRARuntime`.
4. All registered LoRAs are CPU resident after registration succeeds; GPU residency is opportunistic and slot-based.
5. `unregister_lora(name)` means "disable new requests, do not kill old requests".
6. Scheduler admission must consider LoRA GPU slot capacity in addition to normal KV-cache and batch limits.
7. Runtime execution must translate request-level `adapter_id` into rank-local `slot_id` immediately before model execution.
8. LoRA runtime metadata and buffers must remain CUDA-graph-friendly: fixed-capacity graph-visible tensors, fixed slot buffers, and content-only updates during replay.

## Detailed Constraints

### 1. Public API Ownership

#### `ServerPool`

`ServerPool` is the only public truth source for LoRA name visibility.

- `nanovllm_voxcpm/models/voxcpm/server.py:574` rejects duplicate registration names before issuing per-server registration.
- `nanovllm_voxcpm/models/voxcpm/server.py:578` to `nanovllm_voxcpm/models/voxcpm/server.py:589` commit registration only after all servers succeed, and roll back already-registered servers on failure.
- `nanovllm_voxcpm/models/voxcpm/server.py:592` to `nanovllm_voxcpm/models/voxcpm/server.py:606` remove a LoRA from new-request visibility by moving it through `_draining_loras` before final removal from `_registered_loras`.
- `nanovllm_voxcpm/models/voxcpm/server.py:677` rejects `generate(..., lora_name=...)` when the name is absent or draining.

The same semantics are mirrored in `nanovllm_voxcpm/models/voxcpm2/server.py:496` to `nanovllm_voxcpm/models/voxcpm2/server.py:555`.

#### `Server`

`Server` is a transport and state-replica layer only.

- `nanovllm_voxcpm/models/voxcpm/server.py:162` to `nanovllm_voxcpm/models/voxcpm/server.py:171` forward register/unregister/list calls to the local engine.
- `Server` does not retain long-term LoRA payload ownership beyond what the local engine already owns.

### 2. Engine Control Plane

`LoRAManager` is the authoritative local control-plane state machine.

- `nanovllm_voxcpm/engine/lora_manager.py:229` keeps `LoRAManager` focused on registration, lifecycle state, adapter identity, and admission semantics.
- `nanovllm_voxcpm/engine/llm_engine.py:136` to `nanovllm_voxcpm/engine/llm_engine.py:183` fan out register/unregister and sequence lifecycle events to both the engine manager and the runner runtime.

The engine-side manager owns:

- `name -> adapter_id`
- stable `adapter_id` allocation
- lifecycle state: `REGISTERED`, `ACTIVE`, `DRAINING`, `REMOVED`
- CPU-side request reference count (`cpu_ref_count`)
- running-sequence reference count (`gpu_running_ref_count`) for scheduling safety
- `resolve_lora()` admission for new requests
- `can_schedule()` capacity checks for the scheduler

The engine-side manager must not own:

- per-rank payload sharding semantics
- GPU weight buffers
- direct H2D copies
- per-layer slot loading
- token-to-slot execution metadata

### 3. Runner Runtime Plane

`LoRARuntime` is the authoritative local execution/runtime state machine.

- `nanovllm_voxcpm/engine/lora_manager.py:245` keeps `LoRARuntime` responsible for runtime payload retention, slot assignment, LRU eviction, and batch-plan construction.
- `nanovllm_voxcpm/engine/model_runner.py:249` to `nanovllm_voxcpm/engine/model_runner.py:304` route all runner-side LoRA operations through `lora_runtime`.

The runner runtime owns:

- rank-local `LoRAModelPayload`
- `adapter_id -> slot_id | None`
- `slot_id -> adapter_id | None`
- slot resident state: `ACTIVE` or `IDLE`
- slot `last_used_ts` for LRU eviction
- `build_batch_plan()` for execution-time mapping
- `_load_lora_slot()` for H2D load into LoRA-capable modules

The runtime layer is explicitly:

- per-runner
- per-rank
- execution-oriented
- not the public truth source for LoRA registration visibility

### 4. CPU Residency Rules

After successful registration, LoRA payloads are CPU resident and independent from the original checkpoint path.

- `nanovllm_voxcpm/models/voxcpm/engine.py:46` loads the checkpoint into model payloads before delegating to `LLMEngineBase.register_lora()`.
- `nanovllm_voxcpm/engine/llm_engine.py:136` validates and registers the payload across runner ranks, then stores the local adapter identity in the engine manager.
- `nanovllm_voxcpm/engine/model_runner.py:249` stores each rank-local payload in `LoRARuntime`.

This means:

- registration performs loading, parsing, and validation up front
- runtime scheduling no longer depends on the checkpoint directory path
- CPU-resident LoRA metadata outlives GPU residency

### 5. Lifecycle State Machine

The lifecycle is defined by `nanovllm_voxcpm/engine/lora_manager.py:12` and the transition methods in `nanovllm_voxcpm/engine/lora_manager.py:127` to `nanovllm_voxcpm/engine/lora_manager.py:188`.

#### States

- `REGISTERED`: adapter is available for new requests, with no bound requests.
- `ACTIVE`: at least one bound request exists.
- `DRAINING`: new requests are rejected, but already-bound requests continue.
- `REMOVED`: local state has been fully released.

#### Request lifecycle transitions

- `on_sequence_enqueued()` increments `cpu_ref_count` and moves `REGISTERED -> ACTIVE`.
- `on_sequence_started()` increments `gpu_running_ref_count` and refreshes slot protection state.
- `on_sequence_preempted()` decrements `gpu_running_ref_count` without dropping CPU ownership.
- `on_sequence_finished()` decrements the running count when needed, decrements CPU ownership, and either:
  - removes the entry when `state == DRAINING` and `cpu_ref_count == 0`, or
  - returns to `REGISTERED` when `cpu_ref_count == 0` and the adapter is not draining.

#### `unregister_lora(name)` semantics

- `nanovllm_voxcpm/engine/lora_manager.py:124` rejects new requests once the state becomes `DRAINING`.
- `nanovllm_voxcpm/engine/lora_manager.py:134` to `nanovllm_voxcpm/engine/lora_manager.py:139` move the entry to `DRAINING`, then remove it immediately only if no bound request remains.

This repository therefore defines `unregister_lora(name)` as:

- remove the LoRA from future request admission immediately
- allow already-bound waiting or running sequences to continue normally
- release CPU and GPU state only after the final old request leaves the lifecycle

### 6. GPU Residency and Eviction Rules

GPU residency is managed as a fixed slot pool with LRU eviction of idle entries only.

- `nanovllm_voxcpm/engine/lora_manager.py:57` defines `LoRASlot` as `slot_id`, `adapter_id`, `resident_state`, and `last_used_ts`.
- `nanovllm_voxcpm/engine/lora_manager.py:332` to `nanovllm_voxcpm/engine/lora_manager.py:370` implement slot admission and eviction.

The runtime must follow these rules:

- if an adapter is already resident, reuse its slot
- if an empty slot exists, load the adapter there
- otherwise, choose the LRU `IDLE` victim not needed by the current batch
- never evict an `ACTIVE` adapter
- GPU eviction removes only runtime residency, not CPU registration state

`gpu_running_ref_count` is the safety signal for whether an adapter is protected from eviction.

### 7. Scheduling Constraint

LoRA affects scheduling before the runner executes a step. In the current split, the engine performs a conservative control-plane capacity check, while the runner performs the concrete slot assignment and LRU eviction during batch planning.

- `nanovllm_voxcpm/engine/llm_engine.py:166` delegates scheduler admission checks to `LoRAManager.can_schedule()`.
- `nanovllm_voxcpm/engine/lora_manager.py:190` to `nanovllm_voxcpm/engine/lora_manager.py:205` treat LoRA slot availability as a hard scheduling resource.

For any candidate admission, the engine-side admission check must conservatively consider:

- the adapters already represented by running sequences
- the candidate adapter introduced by the waiting sequence
- which adapters are effectively protected by current running state
- whether the required distinct adapter set can fit within the configured LoRA slot capacity

If the required adapters cannot fit, the sequence remains waiting. The runner-side runtime then performs the exact resident-slot reuse, empty-slot fill, or idle LRU eviction when building the execution batch plan.

### 8. Runtime Mapping and Layer Contract

The execution contract is `request adapter binding -> batch slot mapping -> layer-local LoRA kernel metadata`.

- `nanovllm_voxcpm/engine/model_runner.py:298` to `nanovllm_voxcpm/engine/model_runner.py:334` build `token_to_slot`, grouped token indices, active slot ids, slot offsets, and scratch space.
- `nanovllm_voxcpm/utils/context.py:18` to `nanovllm_voxcpm/utils/context.py:87` keep LoRA runtime metadata separate from attention metadata.
- `nanovllm_voxcpm/layers/lora.py:146` to `nanovllm_voxcpm/layers/lora.py:160` package LoRA runtime metadata for backend kernels.

This means:

- `adapter_id` is stable request identity
- `slot_id` is transient local execution placement
- `slot = -1` means no LoRA for that token
- mixed LoRA and non-LoRA batches must stay in one batch and rely on mapping, not batch splitting

### 9. CUDA and CUDA Graph Constraints

LoRA execution must remain graph-safe and runner-owned.

- `nanovllm_voxcpm/engine/model_runner.py:555` to `nanovllm_voxcpm/engine/model_runner.py:686` preallocate fixed-capacity graph-visible LoRA tensors for decode graph capture.
- `nanovllm_voxcpm/engine/model_runner.py:703` to `nanovllm_voxcpm/engine/model_runner.py:760` replay graphs by mutating contents of existing buffers rather than allocating new graph-visible tensors.
- `nanovllm_voxcpm/engine/model_runner.py:277` to `nanovllm_voxcpm/engine/model_runner.py:296` perform LoRA slot H2D transfer only inside the runner.

Required runtime properties:

- `max_loras` is fixed for a runner instance
- `max_lora_rank` is fixed for a runner instance
- graph-visible LoRA metadata and scratch tensors have stable backing storage
- slot weight buffers are reused in place
- graph replay updates buffer contents only

### 10. Tensor Parallel Rules

LoRA registration is consistent by adapter identity, while runtime residency is rank-local.

- `nanovllm_voxcpm/engine/llm_engine.py:139` allocates a stable adapter id once, then broadcasts registration to all ranks.
- `nanovllm_voxcpm/engine/model_runner.py:99` selects rank-local payload shards during runner registration.

The required invariant is:

- same request across ranks -> same logical `adapter_id`

The intentionally local details are:

- each rank may hold different payload shards
- each rank may map the adapter to a different `slot_id`
- each rank manages its own idle eviction and runtime metadata

LoRA must not introduce a new tensor-parallel communication pattern beyond the base layer semantics already used by the model.

## Direct Consequences

### Benefits

- The code now has an explicit architectural split between control-plane and runtime-plane LoRA state.
- Draining semantics are easier to reason about and test.
- CPU registration and GPU residency can evolve independently.
- Scheduler logic can reason about LoRA capacity without owning payloads.
- Runner and layer code can optimize slot residency and CUDA graph replay without leaking those details upward.

### Costs

- LoRA logic is spread across `ServerPool`, engine, runner, and layers, so documentation and tests must preserve the boundaries.
- Lifecycle events must be mirrored intentionally from engine to runner.
- Compatibility is broken for any out-of-tree code that expected runner-side `lora_manager` instead of `lora_runtime`.

## Repository Rules

### Required

- Public APIs expose LoRAs by `name`, not by `adapter_id`.
- Engine code may reason about `adapter_id`, lifecycle, and admission, but not about layer-local slot buffers.
- Runner code owns LoRA payload materialization, H2D slot loads, and graph-visible runtime metadata.
- Layer code must consume LoRA metadata only through the LoRA context.
- A draining LoRA must reject new requests while allowing already-bound requests to finish.

### Known Current Boundaries

- `ServerPool` currently executes LoRA register/unregister serially across servers, even though the public semantic is still all-success-before-commit.
- If pool-level `unregister_lora()` fails partway through, the current implementation can leave the pool in `_draining_loras` while individual servers may have diverged. This is a known operational edge case; future changes should add rollback, reconciliation, or explicit failed-drain recovery semantics.
- The current LoRA context implementation replaces the context object in `set_lora_context()` rather than mutating a persistent singleton in place. Graph replay remains safe because replay uses fixed graph-visible tensors, but future refactors should preserve stable tensor identity for graph-visible buffers.

## Review Checklist

Any of the following changes should be treated as ADR-sensitive:

- merging engine lifecycle state back into runner runtime state
- letting the scheduler inspect per-layer or per-slot GPU buffers directly
- storing public registration truth in a single server instead of the pool
- allowing `unregister_lora()` to cancel or strand existing waiting/running requests
- replacing fixed slot identity with dynamic per-step tensor allocation
- moving LoRA H2D loading out of the runner

## Alternatives Considered

### Option A: Keep one shared manager type for both engine and runner

Rejected. It obscures the control-plane/runtime-plane boundary and encourages future coupling between scheduler policy and execution residency.

### Option B: Let the engine own GPU slot loading directly

Rejected. That would violate the repository's runner-owned device boundary and make CUDA graph constraints harder to reason about.

### Option C: Make GPU residency part of public registration state

Rejected. GPU slots are transient execution details and must remain local to a runner rank.

## Follow-up Actions

1. Add a companion standard under `docs/standards/` if we want code-review rules for LoRA lifecycle changes.
2. Add lifecycle diagrams if future LoRA transitions become more complex.
3. Keep control-plane tests and runtime-plane tests separate when adding new LoRA behavior.
