# LoRA Refactor Plan

## Public API (Server / ServerPool)

本阶段只确定公开接口，不涉及内部实现细节，也不保留任何旧 LoRA 接口兼容。

### Breaking Change Statement

- 本次 LoRA 改造是一次破坏性重构。
- 旧的 LoRA 公开接口不会保留，也不会提供兼容层。
- 以下旧接口全部视为移除，不再对外支持：
  - `load_lora`
  - `reset_lora`
  - `set_lora_enabled`
- 新方案下，LoRA 的唯一公开使用方式是：
  - 先通过 `register_lora(name, path)` 注册
  - 再通过 `generate(..., lora_name=...)` 在请求级选择
- 任何依赖旧 LoRA 接口的调用方都需要按新接口重写。

### 设计原则

- 公开接口只存在于 `ServerPool / Server` 层。
- 公开接口只暴露 `name`，不暴露内部 `adapter_id`。
- 请求侧通过 `lora_name` 选择 LoRA；当 `lora_name is None` 时表示不使用 LoRA。
- LoRA 的注册状态由 `ServerPool` 统一维护，并保证内部所有 `Server` 的注册状态完全一致。
- 不保留旧接口：`load_lora`、`reset_lora`、`set_lora_enabled` 全部移除。

### LoRA Artifact Layout

公开接口对应的 LoRA 输入对象是一个 checkpoint 目录；注册时的 `path` 应指向该目录。

当前需要明确支持的目录结构如下：

```text
<lora_checkpoint_dir>/
├── lora_weights.safetensors
└── lora_config.json
```

注册接口的 `path` 约定：

- `path` 必须指向一个具体可加载的 LoRA checkpoint 目录。
- 该目录下只包含并要求以下两个文件：
  - `lora_weights.safetensors`
  - `lora_config.json`

当前文档层面的格式要求：

- 必需文件：`lora_weights.safetensors`
- 必需文件：`lora_config.json`
- `lora_config.json` 中应包含 `lora_config` 对象

本次重构以 safetensors LoRA 目录格式为主目标；公开接口层不再围绕旧的 `.ckpt` LoRA 接口做兼容设计。

### Public Methods

公开接口收敛为以下四个方法：

- `register_lora(name: str, path: str) -> RegisterLoRAResponse`
- `unregister_lora(name: str) -> UnregisterLoRAResponse`
- `list_loras() -> list[LoRAInfo]`
- `generate(..., lora_name: str | None = None)`

同步与异步封装层保持同名同语义。

### 接口语义

#### `register_lora(name, path)`

- 向当前 `ServerPool` 注册一个可供后续请求使用的 LoRA。
- `name` 是唯一公开标识。
- `path` 只在注册时提供，后续推理请求不再传入路径。
- 若 `name` 已存在，应直接报错，不做覆盖。
- 只有当池内所有 `Server` 都完成注册后，`ServerPool` 才提交该注册状态。

#### `unregister_lora(name)`

- 从当前 `ServerPool` 中移除一个已注册 LoRA。
- 若 `name` 不存在，应直接报错。
- 只有当池内所有 `Server` 都完成卸载后，`ServerPool` 才提交该状态变更。

#### `list_loras()`

- 返回当前 `ServerPool` 已注册的 LoRA 列表。
- 返回结果以 `ServerPool` 自身维护的注册表为准。
- 不依赖某个单独 `Server` 的即时查询结果作为真相源。

#### `generate(..., lora_name=None)`

- 当 `lora_name is None` 时，请求走 base model，不使用 LoRA。
- 当 `lora_name` 为字符串时，请求使用该名称对应的已注册 LoRA。
- 若 `lora_name` 未注册，应直接报错。
- `generate` 不接受 LoRA 路径，不隐式注册 LoRA。

### 状态一致性要求

- `ServerPool` 是公开状态的唯一真相源。
- 所有 `Server` 必须拥有完全一致的 LoRA 注册状态。
- `register_lora` / `unregister_lora` 必须采用“全成功才提交”的语义。
- 若任一 `Server` 操作失败，则整个 `ServerPool` 操作失败，并进入回滚或错误上抛流程。
- `generate(..., lora_name=...)` 只允许引用 `ServerPool` 注册表中存在的名称。
- `Server` 层按串行方式执行相关控制操作；当前设计不要求额外定义 `register/unregister/generate/list` 的并发控制语义。

### 当前阶段边界

- 当前文档只记录 public API 约定。
- `adapter_id`、LoRA 缓存、LoRA 加载器、LoRA manager、batch mapping、layer 执行方式均属于内部设计，后续单独追加。

## Public API Implementation by Layer

本节按 `ServerPool -> Server -> Engine` 三层说明 public API 的实现职责。

### ServerPool

- `ServerPool` 是唯一公开入口。
- `ServerPool` 只维护 LoRA 的注册状态，不长期保存 `path`。
- `ServerPool` 的公开接口为：
  - `register_lora(name: str, path: str)`
  - `unregister_lora(name: str)`
  - `list_loras()`
  - `generate(..., lora_name: str | None = None)`
- `ServerPool` 只负责：
  - 校验 LoRA 名称是否存在或重复
  - 校验注册时提供的目录是否合法
  - 维护全局已注册 `name` 集合
  - 保证所有 `Server` 的 LoRA 注册状态完全一致
- `register_lora(name, path)` 的流程：
  - `ServerPool` 先校验 `name` 与 `path`
  - 并行调用所有 `Server.register_lora(name, path)`
  - 全部成功后，`ServerPool` 只记录该 `name` 已注册
- `unregister_lora(name)` 的流程：
  - `ServerPool` 先校验该 `name` 已存在
  - 并行调用所有 `Server.unregister_lora(name)`
  - `ServerPool` 立即将该 `name` 从可接收新请求的注册集合中移除
  - 已经进入系统并绑定该 LoRA 的请求允许继续执行直到完成
- `generate(..., lora_name)` 的流程：
  - `ServerPool` 快速校验 `lora_name` 是否已注册
  - 若 `lora_name is None`，则表示本次请求不使用 LoRA
  - 选定目标 `Server` 后，将 `lora_name` 原样下发

### Server

- `Server` 不是公开状态真相源，而是 `ServerPool` 的状态副本。
- `Server` 层同样使用 `str name` 标识 LoRA。
- `Server` 不负责长期持有 LoRA 权重内容。
- `Server` 只负责：
  - 接收 `ServerPool` 下发的控制命令
  - 将 `register/unregister/generate` 请求转发给本机 `Engine`
  - 维护最小的本机注册状态副本以配合 public API 语义
- `register_lora(name, path)` 的职责：
  - 将 `name` 与 `path` 转交给本机 `Engine` / `LoRA Manager`
- `unregister_lora(name)` 的职责：
  - 将禁新不杀旧语义转交给本机 `Engine` / `LoRA Manager`
- `generate(..., lora_name)` 的职责：
  - 接收来自 `ServerPool` 的 `lora_name`
  - 正常情况下，本机应已注册该 `name`
  - 将 `lora_name` 原样透传到 `Engine`

### Engine

- `Engine` 不暴露任何 public LoRA API。
- `Engine` 接收 `lora_name: str | None`，而不是公开路径，也不是全局统一 id。
- `Engine` 负责将 `lora_name` 转换为本机内部执行使用的 `internal_id`。
- `Engine` 内部维护本机局部映射：
  - `name -> internal_id`
- `add_request(..., lora_name)` 的职责：
  - `lora_name is None` 时，映射到 no-lora
  - `lora_name` 为字符串时，解析到本机内部 `internal_id`
  - 将该 `internal_id` 写入请求级 payload
- 不要求不同 `Server` 上同一个 `name` 对应相同的 `internal_id`。

### Key Constraints

- `path` 只在 `register_lora(name, path)` 时使用一次。
- 注册完成后，系统运行不再依赖 LoRA 原始目录路径。
- `ServerPool` 只保证所有 `Server` 拥有相同的 LoRA 注册名称集合。
- LoRA 的长期数据由 `Engine` 层的 `LoRA Manager` 统一负责。
- `Server` 不长期管理 LoRA 权重内容。
- `Engine` 负责把 `name` 转换为本机内部执行标识。

## Engine-side LoRA Lifecycle

`Engine` 层将 LoRA 视为本机运行时资源，而不是 public 配置项。

### Engine Responsibilities

- `Engine` 接收 `lora_name: str | None`
- `Engine` 负责将 `lora_name` 转换为本机内部 `internal_id`
- `Engine` 维护本机局部映射：
  - `name -> internal_id`
  - `internal_id -> runtime handle`
  - `name -> cpu_ref_count`
  - `name -> gpu_running_ref_count`
  - `name -> lifecycle state`

### Lifecycle States

- `REGISTERED`
  - LoRA 已在本机注册，可被新请求使用
- `ACTIVE`
  - 至少有一个运行中请求正在使用该 LoRA
- `DRAINING`
  - 已执行 `unregister_lora(name)`，禁止新请求使用
  - 但已有旧请求仍可继续执行
- `REMOVED`
  - 没有任何运行中请求引用该 LoRA
  - 对应运行时资源已释放

### Lifecycle Rules

- 注册完成后，LoRA 进入 `REGISTERED`
- 新请求绑定该 LoRA 后：
  - `cpu_ref_count += 1`
  - 状态进入或保持 `ACTIVE`
- 执行 `unregister_lora(name)` 后：
  - 该 LoRA 不再接受新请求
  - 若 `cpu_ref_count > 0`，状态进入 `DRAINING`
  - 若 `cpu_ref_count == 0`，可直接进入 `REMOVED`
- 当旧请求结束时：
  - `cpu_ref_count -= 1`
  - 若状态为 `DRAINING` 且 `cpu_ref_count == 0`，则进入 `REMOVED`
  - 若状态为 `ACTIVE` 且 `cpu_ref_count == 0`，则回到 `REGISTERED`

### Reference Counting Semantics

- CPU 使用计数：表示当前还有多少个 seq 位于 `waiting` 或 `running` 队列中，并绑定该 LoRA。
- GPU slot 使用计数：表示当前有多少个 seq 处于 `running` 状态并使用该 LoRA。
- CPU 使用计数决定该 LoRA 是否还能被彻底释放。
- GPU slot 使用计数决定该 LoRA 是否允许从 GPU 缓存池中被驱逐。

### Semantic Rule

- `unregister_lora(name)` 的最终语义是：禁新不杀旧。
- 即：
  - 后续新请求不能再使用该 LoRA
  - 已经开始执行并绑定该 LoRA 的旧请求允许自然完成

### Detailed Semantics of "No New, Do Not Kill Old"

- `unregister_lora(name)` 一旦返回成功，该 `name` 必须立即对后续新请求不可见。
- 所谓“新请求”，指的是在 `unregister_lora(name)` 生效之后才进入 `ServerPool.generate(..., lora_name=name)` 校验路径的请求。
- 所谓“旧请求”，指的是在 `unregister_lora(name)` 生效之前，已经完成 LoRA 绑定并进入本机 `Engine` 生命周期管理的请求。
- 旧请求无论当时处于 `waiting` 还是 `running`，都必须允许继续执行直到正常完成、取消或失败退出。
- `unregister_lora(name)` 不能主动取消旧请求，不能把旧请求从 `waiting` 队列中剔除，也不能因为该 LoRA 进入 `DRAINING` 而阻止这些旧请求后续从 `waiting` 进入 `running`。
- `DRAINING` 的含义是：
  - 禁止新的 seq 绑定该 LoRA
  - 允许已经绑定该 LoRA 的旧 seq 继续参与调度和执行
- 只有当最后一个绑定该 LoRA 的旧 seq 生命周期结束后，才允许真正释放该 LoRA 的 CPU 常驻内容与 GPU 缓存内容。
- 若某个 LoRA 已处于 `DRAINING`，再次调用 `unregister_lora(name)` 不应改变既有旧请求的行为；其效果应视为重复执行同一“禁新”动作。

## Model-specific LoRA Design Principle

本次重构不在当前阶段提前固化过细的模型内部接口与函数签名，但需要明确以下设计原则。

### Core Principle

- LoRA 的公共能力与模型相关能力必须分层。
- 公共层负责 LoRA 的注册、卸载、生命周期、请求绑定和多 `Server` 状态一致性。
- 模型层负责 LoRA 的具体配置解释、权重解析、key 映射、shape 校验以及运行时装配。

### What Belongs to the Common Layer

- `ServerPool / Server / Engine` 三层的 public API 语义
- LoRA 注册状态管理
- `unregister_lora(name)` 的禁新不杀旧语义
- 请求级 `lora_name` 绑定
- 本机 `name -> internal_id` 生命周期管理
- 引用计数与 `REGISTERED / ACTIVE / DRAINING / REMOVED` 状态流转

### What Belongs to the Model Layer

- `lora_config.json` 的具体字段解释
- `lora_weights.safetensors` 的具体 key 组织形式
- 原始 key 到模型内部参数名的映射规则
- 每个 LoRA tensor 的 shape 校验规则
- 针对目标模型结构的权重装配、打包或运行时预处理方式

### Intended Architecture Direction

- 公共层不应假设所有模型共享同一种 LoRA key 命名或相同 tensor 结构。
- 公共层只需要知道“某个名称对应一个已加载的 LoRA 运行时对象”。
- 每个模型可以在自己的模块内部实现其专属的 LoRA 解析与运行时构建逻辑。
- 后续新增模型时，应复用公共层的注册与生命周期机制，而不是复制一套新的 public API。

### Explicit Non-goal for This Stage

- 当前阶段不要求提前确定所有模型内部 LoRA 适配接口的最终函数签名。
- 当前阶段只要求明确：模型相关 LoRA 逻辑必须下沉到各自 model 实现，不能泄漏到公共 public API 设计中。

## Engine-side LoRA Manager

LoRA 的常驻内存管理、GPU 缓存管理、生命周期管理与调度约束统一收敛到 `Engine` 层内部的 `LoRA Manager`。

### Placement

- `LoRA Manager` 属于 `Engine` 层内部组件。
- `LoRA Manager` 是一个应当可在不同 model 间复用的公共模块。
- `ServerPool` 不管理 LoRA 权重内容。
- `Server` 不长期管理 LoRA 权重内容。
- `Engine` 通过 `LoRA Manager` 统一管理 LoRA 的 CPU 常驻、GPU 缓存、生命周期与调度协作。

### CPU Residency

- 所有 LoRA 在 `register_lora(name, path)` 成功后就常驻内存。
- 注册完成后，系统运行不再依赖磁盘路径。
- LoRA 的配置、权重及必要预处理结果都由 `LoRA Manager` 保存在本机内存中。

### GPU Cache Pool

- `LoRA Manager` 维护 GPU LoRA 缓存池。
- GPU 缓存池只缓存 LoRA 的 GPU 运行态，不缓存 public 注册状态。
- GPU 缓存池采用 LRU 管理空闲 LoRA。
- 所有正在使用中的 LoRA 必须始终保留，不能被驱逐。

### GPU Residency States

每个已注册 LoRA 在 GPU 缓存池中的状态至少需要区分：

- `ACTIVE`
  - 当前至少有一个运行中的 seq 正在使用该 LoRA
  - 不允许被驱逐
- `IDLE`
  - 当前没有运行中的 seq 使用该 LoRA
  - 可以被 LRU 驱逐

### Lifecycle and Safety

- `LoRA Manager` 需要维护 LoRA 的生命周期状态与引用计数。
- 必须保证：
  - 正在被请求使用的 LoRA 不会被释放
  - 正在 GPU 上被运行中 seq 使用的 LoRA 不会被驱逐
- `unregister_lora(name)` 后：
  - 禁止新请求继续使用该 LoRA
  - 已绑定该 LoRA 的旧请求继续执行
  - 待引用清零后，`LoRA Manager` 再释放其 CPU 常驻内容与 GPU 缓存内容

### Scheduling Constraint

- 调度不只受 KV cache 和 batch 大小约束，也受 GPU LoRA 缓存池容量约束。
- 在将新的 seq 从 `waiting` 调度到 `running` 前，`Engine` 必须通过 `LoRA Manager` 判断：
  - 该批次涉及的 LoRA 是否已经驻留在 GPU 缓存池中
  - 若未驻留，是否可以通过驱逐 `IDLE` LoRA 腾出空间
  - 若 GPU LoRA 缓存池无法容纳该批次所需的 LoRA，则不能继续调度更多 seq 进入 `running`

### Scheduling Cooperation

- `LoRA Manager` 需要向 `Engine` 提供调度所需的容量判定能力。
- 对任意候选 seq 集合，`LoRA Manager` 至少能够判断：
  - 当前 `running` 集合涉及哪些 LoRA
  - 候选 seq 会新增哪些 LoRA
  - 这些新增 LoRA 是否已经 GPU resident
  - 若尚未 resident，是否可以通过驱逐 `IDLE` LoRA 腾出足够 slot
- `Engine` 只有在 `LoRA Manager` 判定 GPU 缓存池可满足时，才允许将新的 seq 送入 `running`。
- 若 GPU 缓存池无法满足，则该 seq 保持在 `waiting`，等待后续调度周期。

### Intended Effect

- 已注册 LoRA：始终 CPU 常驻
- 运行中需要的 LoRA：GPU 上保持 `ACTIVE`
- 暂时不用的 LoRA：GPU 上可保留为 `IDLE`，并由 LRU 管理
- 当 GPU LoRA 缓存池放不下时，调度侧必须主动收敛，不引入更多需要新 LoRA 的运行中 seq

## LoRA Cache Pool Implementation Details

本节细化 `Engine` 层 `LoRA Manager` 的缓存池实现方式。整体思路参考 vLLM per-request LoRA：CPU registered adapters 与 GPU active adapters 分离，GPU 使用固定 slot pool，执行前将请求级 adapter 映射到 GPU slot。

### Cache Levels

`LoRA Manager` 维护两级缓存：

- `Registered Cache`
  - CPU 常驻缓存
  - 保存所有已注册 LoRA
  - 保存解析后的配置、映射后的权重、运行时元数据
- `Active Cache Pool`
  - GPU LoRA 缓存池
  - 固定大小为 `max_loras`
  - 只保存当前驻留 GPU 的 LoRA
  - 使用 LRU 管理空闲 LoRA
  - 正在使用的 LoRA 绝不驱逐

### Identity and Mapping

- 每个 LoRA 在本机 `Engine` 内有稳定的 `adapter_id`。
- GPU 上的 `slot_id` 是临时驻留位置。
- `adapter_id` 与 `slot_id` 必须分离。
- `LoRA Manager` 至少维护以下映射：
  - `name -> adapter_id`
  - `adapter_id -> cpu_entry`
  - `adapter_id -> slot_id | None`
  - `slot_id -> adapter_id | None`

### CPU Entry Metadata

每个 CPU 常驻 LoRA entry 至少包含：

- `name`
- `adapter_id`
- `state`
- `ref_count`
- `rank`
- `alpha`
- `scaling = alpha / rank`
- `model_payload`
- `gpu_resident`
- `gpu_active_count`
- `last_used_ts`

其中：

- `model_payload` 是模型相关对象，包含各模型自行定义的 LoRA 配置、权重、key 映射结果、shape 校验结果或预处理结果。
- `ref_count` 表示仍有多少未完成请求绑定该 LoRA。
- `gpu_active_count` 表示当前有多少 running seq 或当前 step 正在使用该 LoRA。

### GPU Slot Metadata

GPU 缓存池是固定大小的 slot pool。每个 slot 至少包含：

- `slot_id`
- `adapter_id | None`
- `resident_state`
- `last_used_ts`
- 各 LoRA-capable layer 中该 slot 对应的 LoRA 权重视图

`resident_state` 至少区分：

- `ACTIVE`
  - 当前至少有一个 running seq 或当前 step 正在使用该 LoRA
  - 不允许被驱逐
- `IDLE`
  - 当前驻留 GPU，但没有 running seq 或当前 step 使用
  - 可以被 LRU 驱逐

### Register Flow

- `register_lora(name, path)` 成功后，LoRA 进入 CPU 常驻缓存。
- 注册流程需要完成：
  - 读取 `lora_config.json`
  - 读取 `lora_weights.safetensors`
  - 执行模型相关配置解析、key 映射、shape 校验与必要预处理
  - 分配本机 `adapter_id`
  - 写入 `Registered Cache`
- 注册时不要求立即进入 GPU 缓存池。
- 注册完成后，不再依赖磁盘路径。

### Unregister Flow

- `unregister_lora(name)` 的语义仍然是禁新不杀旧。
- 执行卸载后：
  - 禁止新请求继续绑定该 LoRA
  - 已绑定该 LoRA 的旧请求继续执行
  - 若 `ref_count > 0`，保留 CPU 常驻内容以及必要的 GPU resident 内容
  - 若 `ref_count == 0`，释放 CPU entry，并在 GPU slot 处于 `IDLE` 时释放 GPU resident 内容
- 若 LoRA 处于 `DRAINING` 且最后一个旧请求结束，则彻底删除其 CPU 与 GPU 缓存状态。

### Request Lifecycle

- `Engine.add_request(..., lora_name)` 时：
  - `LoRA Manager` 将 `lora_name` 解析为本机 `adapter_id`
  - `cpu_ref_count += 1`
  - 状态进入或保持 `ACTIVE`
- 当 seq 从 `waiting` 进入 `running` 时：
  - `gpu_running_ref_count += 1`
- 请求 `finish` 或 `cancel` 时：
  - 若 seq 当时处于 `running`，则 `gpu_running_ref_count -= 1`
  - `cpu_ref_count -= 1`
  - 若状态为 `DRAINING` 且 `cpu_ref_count == 0`，释放该 LoRA
  - 若状态不是 `DRAINING` 且 `cpu_ref_count == 0`，回到 `REGISTERED`

### GPU Admission and Eviction

当某个 step 需要 `adapter_id` 驻留 GPU 时：

- 若该 adapter 已在 GPU 缓存池中：
  - 直接复用已有 slot
  - 标记为 `ACTIVE`
  - 更新 `last_used_ts`
- 若该 adapter 不在 GPU 缓存池中：
  - 若存在空 slot，直接装入
  - 若不存在空 slot，则从 `IDLE` resident 中按 LRU 选择 victim
  - 若没有任何 `IDLE` slot，则该 step 不能接纳该 adapter

驱逐规则：

- 只能驱逐 `IDLE` LoRA。
- 不能驱逐当前 step 或 running seq 正在使用的 `ACTIVE` LoRA。
- 被驱逐的 LoRA 只移出 GPU 缓存池，不影响 CPU 常驻注册状态。

装载细节：

- GPU slot 的装载只发生在 runner。
- GPU slot 的 H2D 拷贝使用默认 stream。
- 当前设计假定该装载过程不会失败；若运行环境无法满足，则应在更早阶段暴露初始化或注册错误，而不是在运行时引入额外失败分支。

### Scheduling Constraint

调度时除了 KV cache 与 batch size，还必须考虑 GPU LoRA 缓存池容量。

对候选 batch，需要检查：

- 该 batch 涉及的 distinct `adapter_id` 集合
- 其中哪些 adapter 已经 GPU resident
- 哪些 adapter 需要新占用 GPU slot
- 当前 GPU 缓存池是否能通过空 slot 或驱逐 `IDLE` LoRA 来满足需求

若 GPU LoRA 缓存池无法容纳该候选 batch 所需 LoRA，则不能继续将更多 seq 从 `waiting` 调度到 `running`。

硬约束：

- 一个 step 中需要同时使用的 distinct LoRA 数量不能超过 `max_loras`。
- GPU 缓存池中全是 `ACTIVE` LoRA 时，不能调度会引入新 LoRA 的 seq。

### Rank and Alpha Handling

- 允许不同 LoRA 使用不同的 `rank` 与 `alpha`。
- `LoRA Manager` 需要有统一的 `max_lora_rank`。
- 注册时必须校验：`rank <= max_lora_rank`。
- GPU slot 按 `max_lora_rank` 统一预分配。
- 每个 adapter / slot 单独保存：
  - `effective_rank`
  - `scaling = alpha / rank`
- 小 rank LoRA 只使用前 `effective_rank` 部分；剩余区域可以零填充，或由后续 CUDA kernel 根据 `effective_rank` 跳过。

### Design Notes from vLLM

必须保留的设计思想：

- CPU registered adapters 与 GPU active adapters 分离。
- GPU 使用固定 slot pool，而不是动态散放 LoRA 权重。
- 稳定的 `adapter_id` 与临时的 `slot_id` 分离。
- batch 执行前将 `adapter_id` 翻译为 `slot_id`。
- 调度阶段必须考虑一个 step 内的 distinct LoRA 数量与 GPU slot 容量。

可以针对本仓库改进的点：

- 同时维护 `adapter_id -> slot_id` 与 `slot_id -> adapter_id`，避免执行时线性反查。
- 在调度阶段不仅检查 distinct LoRA 数量，也检查 GPU 缓存池是否可通过 LRU 驱逐 `IDLE` LoRA 满足需求。

## CUDA Kernel and CUDA Graph Design

本节确定 LoRA 推理在 CUDA kernel / CUDA Graph 层面的最终方案。

### LoRA Availability

- 新增类似 `torch.cuda.is_available()` 的可用性接口：`lora.is_available()`。
- `lora.is_available()` 用于快速检查当前运行时是否支持 LoRA 推理。
- 只有当 Punica kernel 与 LoRA 运行时依赖均可用时，`lora.is_available()` 才返回 `True`。
- 若 `lora.is_available()` 返回 `False`：
  - 系统仍可运行 base model
  - 但不允许注册任何 LoRA
- `register_lora(name, path)` 必须再次强校验 `lora.is_available()`；若不可用则直接报错。

### Punica Requirement

- Punica kernel 是 LoRA 推理的硬依赖。
- 不提供非 Punica 的 LoRA fallback 实现。
- 若运行时缺少 Punica kernel，则 LoRA 功能整体不可用。

### Kernel Computation Model

- LoRA 推理采用 vLLM / Punica 风格的 `slot + mapping + shrink/expand` 方案。
- base matmul 只执行一次。
- LoRA 增量通过两阶段 kernel 完成：
  - `shrink`: `tmp = x @ A^T`
  - `expand`: `delta = tmp @ B^T`
- 最终输出形式为：
  - `out = base_out + delta`
- 不拆 batch，不重复 base matmul，不物化 `BA`。

### GPU Weight Layout

- 所有 LoRA-capable layer 使用固定大小的 slot-major GPU 权重布局。
- 每层 LoRA 权重按 slot 存储：
  - `A[slot, max_lora_rank, in]`
  - `B[slot, out, max_lora_rank]`
- `max_loras` 与 `max_lora_rank` 在 `Engine` 初始化时固定。
- 不同 LoRA 可以有不同的 `rank` 与 `alpha`。
- 每个 slot 单独维护：
  - `effective_rank`
  - `scaling = alpha / rank`
- 小 rank LoRA 统一装入 `max_lora_rank` 形状的 slot；仅前 `effective_rank` 部分有效，其余部分零填充或由 kernel 跳过。

### Runtime Mapping Metadata

每个 step 由 runner / `LoRA Manager` 原地更新 LoRA metadata，包括：

- `token_to_slot`
- `token_indices_sorted_by_slot`
- `active_slot_ids`
- `num_tokens_per_slot`
- `slot_start_offsets`
- `no_lora_flag`

约定：

- `slot = -1` 表示 no-LoRA。
- 若整批都没有 LoRA，则走 no-LoRA fast path。
- mixed batch 中 no-LoRA token 只参与 metadata，不参与 LoRA 低秩计算。

### Decode and Prefill

- `decode` 路径采用 row-wise / grouped gather 风格 kernel。
- `prefill` 路径采用按 slot 分组后的 segmented / grouped kernel。
- 两条路径共享同一套 slot-major 权重池与 mapping 设计。

### CUDA Graph Requirements

LoRA 子系统必须对 CUDA Graph 友好，满足以下硬要求：

- `max_loras` 固定
- `max_lora_rank` 固定
- graph 可见的 metadata buffer 固定容量
- graph 可见的 scratch buffer 固定容量
- slot 权重 buffer 固定地址
- metadata tensor 固定地址
- scratch tensor 固定地址
- 运行时只能通过 `copy_` / `fill_` 更新内容，不能替换 graph 可见 tensor 对象

这里所说的 CUDA Graph 硬规格，指的是：

- graph capture 期间可见的 LoRA 相关 tensor 形状必须固定
- graph replay 期间可见的 LoRA 相关 tensor 地址必须固定
- graph replay 时只能修改这些 tensor 的内容，不能替换对象或改变结构
- LoRA 引入后，不能让 graph 的 kernel launch 结构依赖 Python 层动态分支发生变化

当 adapter 进入或替换 GPU slot 时：

- 只能将权重拷贝到既有 slot buffer 中
- 不能新建 graph 可见权重 tensor

### CUDA Graph Dispatch

- CUDA Graph dispatch 至少要区分：
  - `has_lora = false`
  - `has_lora = true`
- 更推荐进一步按 `num_active_loras` 做 bucket，以提高 replay 稳定性。

### Context Design

- 保持现有 attention `get_context / set_context` 语义不变。
- 新增独立的 LoRA runtime context。
- runner 每个 step 同时设置：
  - attention context
  - LoRA context
- LoRA layer 只读取 LoRA context，不复用 attention context 结构。

### Final Principle

- `lora.is_available()` 是 LoRA 能力总开关。
- Punica kernel 是 LoRA 的硬依赖。
- mixed batch 靠 mapping 实现，不靠拆 batch。
- base 路径只计算一次。
- LoRA 只做低秩增量计算。
- CUDA Graph replay 时只改内容，不改结构。

## Tensor Parallel LoRA Design

本节补充 `tensor_parallel_size > 1` 时的 per-request LoRA 设计。

### Core Principle

- TP 下 LoRA 的正确性依赖于 adapter 语义一致，不依赖 `slot_id` 一致。
- 同一个 TP group 内必须保证：
  - 同一个请求绑定的是同一个 LoRA adapter
  - 同一个 step 的 active adapter 集合一致
- 不要求不同 TP rank 上的 `adapter -> slot_id` 完全一致。
- 不要求不同 TP rank 上的 `token_to_slot` 完全一致。

### Minimal-change Strategy

- 每个 `Engine` 持有自己的本机 `LoRA Manager`。
- `register_lora(name, path)` 时即完成：
  - LoRA 解析
  - key 映射
  - shape 校验
  - 按当前 TP rank 完成本地 shard 化
- 注册完成后：
  - 每个 rank 只持有自己的 CPU shard
  - 每个 rank 的 GPU slot pool 只持有自己的 GPU shard
- 推理热路径中不再做 LoRA 分片。

### Adapter and Slot Semantics

- `adapter_id` 应当在同一个 TP group 内具有一致语义。
- `slot_id` 是 rank-local runtime detail，可以在不同 rank 上不同。
- `token_to_slot` 也是 rank-local metadata，只需要在本 rank 上正确指向该 adapter 的本地 slot。
- TP 下必须一致的是：`token -> adapter` 的逻辑绑定。
- TP 下允许不一致的是：`adapter -> local_slot` 的本地实现编号。

### Scheduling and Cache Capacity

- 调度判断基于 distinct adapter 集合，而不是基于 `slot_id` 集合。
- 只要同一个 TP group 内所有 rank 都能为该 batch 需要的 adapter 提供本地 slot，就允许调度。
- 每个 rank 的 GPU cache pool 独立维护自己的：
  - resident state
  - local slot allocation
  - LRU eviction

### Communication Principle

- LoRA 不能引入新的 TP 通信模式。
- LoRA 必须严格继承 base layer 现有 TP 语义：
  - Column / QKV / MergedColumn 类层：LoRA 输出仍是本 rank shard
  - RowParallel 类层：LoRA delta 先加到本地 partial output，再复用原有 `all_reduce`
- 不允许因为 LoRA 额外新增独立的 all-reduce / all-gather。

### CUDA Graph under TP

- CUDA Graph 的 graph-safe 约束在每个 rank 本地分别满足即可。
- 每个 rank 都必须保证：
  - 本地 slot buffer 固定地址
  - 本地 metadata buffer 固定地址
  - 本地 scratch buffer 固定地址
  - replay 时只改内容，不改结构
- 不要求跨 rank 的 `slot_id` 一致，只要求每个 rank 本地 replay 结构稳定。

### Final TP Principle

- TP 下，LoRA 在注册阶段完成 rank-local shard 化。
- 每个 `Engine` 的 `LoRA Manager` 只管理本 rank 的 shard 与本地 slot pool。
- 跨 rank 要求一致的是 adapter 语义，不是 slot 编号。
