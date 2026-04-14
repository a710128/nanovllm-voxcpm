# Per-request LoRA Refactor Implementation Plan

本文件只记录开发实施计划；设计原则与架构决策见 `DESIGN.md`。

## Phase 1: Vendored Triton LoRA Ops 与新 LoRA Layer 基座

目标：先把 LoRA 的 CUDA 执行基座做对，确保后续运行时与调度都建立在稳定的 kernel 语义之上。

当前状态：Phase 1 已实现完成。

### 1.1 引入 LoRA 可用性检查

- 新增 `lora.is_available()` 能力检查接口。
- 检查仓库内 vendored Triton LoRA ops 及其运行时依赖是否可用。
- 明确：若 `lora.is_available() == False`，则系统只能跑 base model，不能注册 LoRA。
- 在注册 LoRA 的代码路径中强制复用该检查。

### 1.2 接入 vendored Triton LoRA ops

- 在仓库内建立 LoRA kernel 适配层，明确 vendored Triton LoRA ops 为唯一 LoRA 后端。
- 文档与实现中将该仓库内实现视为本仓库的 Punica 实现，不再区分外部 Punica backend。
- 统一处理缺失依赖、导入失败、初始化失败的报错路径。
- 不提供其他 LoRA backend 或 fallback。

### 1.3 重构现有 LoRA layer

- 重构 `nanovllm_voxcpm/layers/lora.py` 中现有 LoRA layer，使其支持：
  - slot-major GPU 权重布局
  - fixed `max_loras`
  - fixed `max_lora_rank`
  - `effective_rank`
  - per-slot `scaling`
- 保持 base path 只执行一次。
- LoRA path 改为 Punica 风格的 `shrink + expand` 增量计算，由 vendored Triton LoRA ops 执行。
- 对 fused projection（如 QKV、gate_up）保留 packed/slice 友好的结构。
- 新 LoRA layer 必须同时支持 `tensor_parallel_size > 1`。
- TP 下 LoRA 的张量分片与通信语义必须严格继承 base layer 当前实现。

### 1.4 建立独立 LoRA runtime context

- 保持现有 attention context 不变。
- 新增 LoRA context，用于承载：
  - `token_to_slot`
  - `token_indices_sorted_by_slot`
  - `active_slot_ids`
  - `num_tokens_per_slot`
  - `slot_start_offsets`
  - `no_lora_flag`

### 1.5 保证 CUDA Graph 友好

- 预分配 graph 可见的 LoRA metadata buffer。
- 预分配 graph 可见的 scratch buffer。
- 预分配固定地址的 slot-major GPU 权重池。
- 运行时只允许原地 `copy_` / `fill_`，不替换 graph 可见 tensor。
- 确认 LoRA 打开后不会破坏现有 CUDA graph capture / replay 前提。

### 1.6 Phase 1 验证

- 为新 LoRA layer 补充 CUDA 单测。
- 验证：
  - no-LoRA 与 base 输出一致
  - 单 LoRA 输出正确
  - mixed-LoRA 输出正确
  - 不同 `rank / alpha` 输出正确
  - no-LoRA fast path 正常
  - CUDA Graph capture / replay 不被 break
- 至少补一组 `TP=2` 的 CUDA 单测，覆盖：
  - no-LoRA
  - single-LoRA
  - mixed-LoRA
  - CUDA Graph 不被 break

## Phase 2: Engine 运行时、GPU 缓存池、生命周期与调度

目标：实现可复用的 `LoRA Manager`，把 CPU 常驻、GPU cache pool、生命周期和调度约束全部打通。

当前状态：Phase 2 已实现完成。已具备 Engine 内部 `LoRA Manager`、CPU registered cache、GPU slot pool、生命周期/引用计数、scheduler LoRA capacity admission，以及 runner-owned slot admission / metadata 更新。`ServerPool` public API 与模型侧 checkpoint 解析仍按 Phase 3 执行。

### 2.1 建立通用 LoRA Manager 骨架

- 在 `Engine` 层引入通用 `LoRA Manager` 模块。
- `LoRA Manager` 负责：
  - CPU 常驻 registered cache
  - `name -> internal_id`
  - `internal_id -> runtime handle`
  - lifecycle state
  - `cpu_ref_count`
  - `gpu_running_ref_count`
  - GPU active cache pool

### 2.2 实现 CPU 常驻注册表

- `register_lora(name, path)` 成功后，将 LoRA 解析结果常驻 CPU 内存。
- 注册完成后不再依赖磁盘路径。
- 支持 `REGISTERED / ACTIVE / DRAINING / REMOVED` 生命周期状态。
- 在 `tensor_parallel_size > 1` 时，注册阶段即完成当前 TP rank 的本地 shard 化。
- 每个 rank 的 CPU registered cache 只保存本 rank shard。

### 2.3 实现 GPU slot pool

- 建立固定大小的 GPU slot pool（`max_loras`）。
- 维护：
  - `adapter_id -> slot_id`
  - `slot_id -> adapter_id`
- 每个 slot 支持 `ACTIVE / IDLE` resident 状态。
- 在 `tensor_parallel_size > 1` 时，每个 rank 只维护本地 GPU slot pool。
- 不要求不同 TP rank 上 `slot_id` 完全一致。

### 2.4 实现 LRU 驱逐策略

- 仅允许驱逐 `IDLE` resident LoRA。
- `ACTIVE` LoRA 绝不驱逐。
- 建立 `last_used_ts` 或等价 LRU 元数据。
- 被驱逐的 LoRA 只移出 GPU，不影响 CPU registered cache。

### 2.5 接入精确引用计数

- CPU 使用计数：统计当前还在 `waiting/running` 队列中的 seq。
- GPU 使用计数：统计当前处于 `running` 的 seq。
- 在以下事件中正确维护计数：
  - `add_request`
  - `waiting -> running`
  - `running -> waiting`（若发生）
  - `finish`
  - `cancel`

### 2.6 完善 unregister 语义

- `unregister_lora(name)` 进入 `DRAINING`。
- 新请求立即不可再绑定该 LoRA。
- 已绑定的旧请求继续执行。
- 等 `cpu_ref_count == 0` 后，再彻底释放 CPU / GPU 资源。

### 2.7 接入调度协作

- 在调度从 `waiting` 选入 `running` 前，接入 `LoRA Manager` 的容量判定。
- 需要判断：
  - 当前 batch 的 distinct LoRA 集合
  - 哪些已 resident
  - 哪些需要新 slot
  - 是否可通过驱逐 `IDLE` LoRA 腾出空间
- 若 GPU LoRA 缓存池无法满足，则该 seq 继续留在 `waiting`。
- 在 `tensor_parallel_size > 1` 时，调度判断基于 distinct adapter 集合，而不是基于各 rank 的 `slot_id` 集合。
- 只要同一个 TP group 内所有 rank 都能为所需 adapter 提供本地 slot，才允许调度进入 `running`。

### 2.8 runner 接入 slot admission

- runner 在每个 step 根据 batch 实际需要：
  - 选择 active LoRA 集合
  - 触发 slot admission / eviction
  - 原地更新 LoRA context metadata
- GPU slot 装载固定发生在 runner，使用默认 stream。
- 在 `tensor_parallel_size > 1` 时，每个 rank 独立执行本地 slot admission / eviction。
- `token_to_slot` 与 `slot_id` 允许是 rank-local metadata，只要求跨 rank 的 adapter 语义一致。

### 2.9 Phase 2 验证

- 为 `LoRA Manager` 补单测，覆盖：
  - register / unregister
  - draining 语义
  - CPU / GPU 引用计数
  - slot 分配
  - LRU 驱逐
  - active 不可驱逐
  - 调度容量判定
- 为 scheduler 补 LoRA capacity 相关单测。
- 至少补一组 `TP=2` 的运行时单测，覆盖：
  - rank-local shard 注册
  - 本地 GPU slot pool 管理
  - 基于 adapter 集合的调度容量判定

## Phase 3: 模型接入与 Public API 收口

目标：在运行时能力稳定后，再把 `voxcpm` 接入，并完成 public API 重构。

### 3.1 接入 VoxCPM 模型侧 LoRA 解析

- 在 `voxcpm` 模型侧实现：
  - `lora_config.json` 解析
  - `lora_weights.safetensors` 读取
  - key 映射
  - shape 校验
  - model runtime payload 构建
- 与通用 `LoRA Manager` 对接。
- 在 `tensor_parallel_size > 1` 时，模型侧 LoRA 解析结果需要按当前 TP 规则切分为 rank-local shard payload。

### 3.2 接入 Engine 到 Runner 的 LoRA 数据链路

- 在请求 payload 中接入本机 `internal_id`。
- 让 runner 能基于 payload 中的 LoRA 信息构造 runtime mapping。
- 确认 prefill / decode 两条路径都正确携带 LoRA 信息。

### 3.3 完成 Server / ServerPool 新接口

- 落地以下公开接口：
  - `register_lora(name, path)`
  - `unregister_lora(name)`
  - `list_loras()`
  - `generate(..., lora_name=None)`
- 确保 `ServerPool` 是唯一公开真相源。
- 确保所有 `Server` 的注册状态保持一致。

### 3.4 移除旧 LoRA public 接口

- 删除旧的：
  - `load_lora`
  - `reset_lora`
  - `set_lora_enabled`
- 清理对应示例、测试和调用路径。

### 3.5 Phase 3 验证

- 补充 `voxcpm` 模型侧单测。
- 补充 `ServerPool / Server` public API 单测。
- 补充端到端 mixed-LoRA 推理测试。
- 验证：
  - `register -> generate -> unregister` 主路径正确
  - `unregister` 后旧请求继续完成、新请求被拒绝
  - mixed batch 在真实模型链路中正确执行
- 至少补一组 `TP>1` 的端到端 mixed-LoRA 测试。

## Recommended Execution Order

- 先完成 Phase 1，再进入 Phase 2，最后做 Phase 3。
- 不建议在 vendored Triton LoRA ops 与 CUDA Graph 约束未稳定前，提前改 public API 或模型层调用路径。
- 不建议在 `LoRA Manager` 与调度协作未完成前，提前做端到端接口联调。
- `ServerPool` 的 `register_lora` / `unregister_lora` / `list_loras` / `generate(..., lora_name=...)` 明确属于 Phase 3 范围；在 Phase 3 之前，不要求实现这些 public API，也不要求考虑它们之间的串通、一致性或联调问题。
