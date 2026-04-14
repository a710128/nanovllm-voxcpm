# ADR 0001: Model Runner 独占 Host/Device 传输与同步边界

## 状态

Accepted

## 日期

2026-04-14

## 背景

当前推理链路由 `server -> engine -> scheduler -> model runner -> model -> engine postprocess` 组成。

结合现有实现，推理热路径中的 Host/Device 边界主要位于以下位置：

- `nanovllm_voxcpm/models/voxcpm/server.py:108` 在服务层执行 `wav_tensor.cuda()`。
- `nanovllm_voxcpm/engine/model_runner.py:292` 到 `nanovllm_voxcpm/engine/model_runner.py:360` 在 runner 内构造 attention context 并执行 Host to Device 传输。
- `nanovllm_voxcpm/models/voxcpm/runner.py:125` 到 `nanovllm_voxcpm/models/voxcpm/runner.py:132` 在 runner 内将 payload 从 CPU 异步搬运到 GPU。
- `nanovllm_voxcpm/models/voxcpm/runner.py:151` 到 `nanovllm_voxcpm/models/voxcpm/runner.py:157`、`nanovllm_voxcpm/models/voxcpm/runner.py:170` 在 runner 内将结果从 GPU 取回 CPU。

这意味着当前代码已经大体把数据搬运放在 runner 一侧，但仍存在两个问题：

1. Host/Device 边界没有被正式声明为架构约束。
2. 服务层仍有直接的 `cuda()` 调用，破坏了边界一致性。

与此同时，推理链路对吞吐和尾延迟都敏感。这里的“推理热路径”包含两类执行：

- 逐 step 的生成路径。
- 生成前为 prompt audio 执行的 latent encode 路径。

任何散落在 server、engine、scheduler、model、layer 中的 Host/Device 传输，或者隐式同步，都可能带来以下问题：

- 难以推断一轮 step 的执行顺序与阻塞点。
- 难以保证 CUDA graph / 流式执行的稳定性。
- 容易在看似无害的 `.cpu()`、`.item()`、`.tolist()`、`.numpy()` 中引入同步。
- 让 CPU 校验、GPU 计算、结果回传的职责混杂，降低可维护性。

## 决策

自本 ADR 起，推理热路径遵循以下强制边界：

1. 所有 Host to Device、Device to Host 传输只能发生在 `model runner` 部分。
2. 所有可能触发同步的跨边界读取，也只能发生在 `model runner` 部分。
3. 推理热路径必须遵循严格顺序：
   - 先在 CPU 上完成所有校验与整理。
   - 再由 runner 异步把输入搬运到 GPU。
   - 再在 GPU 上完成全部运算，期间不允许触发任何同步。
   - 最后由 runner 把结果从 GPU 取回 CPU。
4. `server`、`engine`、`scheduler`、`model`、`layers` 不得在推理热路径中直接执行任何跨 Host/Device 边界的操作。

## 具体约束

### 1. 分层职责

#### CPU 边界外层：`server` / `engine` / `scheduler`

只能做以下事情：

- 请求解析、参数校验、长度校验、类型校验。
- Python / NumPy 级别的数据整理。
- `Sequence` / `RunnerTask` 构造与状态推进。
- 调度、KV block 规划、停止条件判断。

不得做以下事情：

- `tensor.cuda()`
- `tensor.to("cuda")`、`module.to("cuda")`
- `torch.tensor(..., device="cuda")`
- `torch.as_tensor(..., device="cuda")`
- `tensor.cpu()`
- `tensor.numpy()`（当 tensor 仍在 GPU 上时）
- `tensor.tolist()`、`tensor.item()`
- `torch.cuda.synchronize()`
- `torch.cuda.current_stream().synchronize()`
- 任何为了读取 GPU 结果而发生的隐式同步

#### GPU 边界层：`BaseModelRunner` 及其子类

负责以下事情：

- 将 CPU 上已经校验完成的数据异步搬运到 GPU。
- 准备 attention context、KV cache 写入位置等 GPU 执行元数据。
- 执行 model forward / CUDA graph replay / VAE decode 等 GPU 计算。
- 在 GPU 计算完成后，统一把结果回传到 CPU。

### 2. 严格时序

每一次推理 step 必须符合以下四阶段，不允许交叉：

#### 阶段 A：CPU 校验与归一化

位置：`server`、`engine`、`scheduler`

要求：

- 完成 shape、dtype、长度、取值范围、请求一致性校验。
- 完成 prompt 拼接、prefix cache 输入整理、停止条件上界计算。
- 形成完整且自洽的 CPU payload。

此阶段结束后，后续不得因为“补校验”再次回到 CPU 读取 GPU 中间结果。

#### 阶段 B：Runner 内异步 H2D

位置：`BaseModelRunner` / `VoxCPMRunner`

要求：

- 必须由 runner 统一发起 H2D。
- 在框架和数据形态允许时，应优先使用 pinned host memory + `non_blocking=True` 执行异步 H2D。
- 一次性准备好本 step 所需的全部 GPU 输入与 context。
- 不允许在 model、attention layer、工具函数中偷偷补做 H2D。

#### 阶段 C：GPU 纯计算

位置：runner 调用 model 之后，到开始 D2H 之前

要求：

- 只允许 GPU kernel、CUDA graph replay、device 上的 tensor 变换。
- 不允许任何会把控制权同步回 CPU 的操作。
- 不允许读取 GPU 标量到 Python。
- 不允许 `.item()`、`.tolist()`、`.cpu()`、`.numpy()`、显式 stream synchronize。

#### 阶段 D：Runner 内统一 D2H

位置：`BaseModelRunner` / `VoxCPMRunner`

要求：

- 只在 GPU 计算全部结束后，把最终需要返回给 engine 的结果搬回 CPU。
- 可以存在多个连续的 D2H 读取，但它们必须全部位于阶段 D，不得与阶段 C 交叉。
- D2H 后得到的对象必须是 engine 可直接消费的 Python / NumPy 数据。
- engine 的 `postprocess_seq()` 不得再访问 GPU tensor。

### 3. 允许的例外

以下例外不属于“推理热路径”，但仍必须留在 runner 层：

- 初始化阶段的分布式同步与 CUDA graph capture。
- 退出阶段的资源回收与同步。

即便属于例外，也不得下沉到 `server`、`engine`、`scheduler`、`model`、`layers`。

## 直接影响

### 正向影响

- Host/Device 边界清晰，便于推理性能分析。
- 更容易发现隐藏同步点。
- 降低 model 和 engine 层被设备语义污染的风险。
- 便于后续为 runner 增加 profiling、stream 管理和传输审计。

### 代价

- runner 职责更重，需要明确承担输入封送和输出解封送。
- 某些历史实现需要从 server 或 engine 回迁到 runner。
- 代码评审需要额外检查“是否越过边界”。

## 本仓库落地规则

### 必须满足

- `Sequence`、`RunnerTask`、`custom_payload` 在进入 runner 前必须是 CPU 可序列化数据。
- `postprocess_seq()` 的输入必须已经是 CPU 数据。
- 所有 GPU tensor 的创建、拷贝、回传都由 runner 统一负责。

### 当前需要对齐的已知位置

- `nanovllm_voxcpm/models/voxcpm/server.py:108` 到 `nanovllm_voxcpm/models/voxcpm/server.py:117`
  当前 `encode_latents()` 在服务层直接把 waveform 放到 GPU。这一类逻辑应迁移为：服务层只做音频解码与校验，runner 独占 H2D 与编码执行。

### 推荐实现模式

- CPU 层：`bytes` / `list` / `np.ndarray` / 标量。
- Runner 入口：集中完成 `torch.from_numpy(...)`、pinned memory、`.cuda(non_blocking=True)`。
- Runner 出口：集中完成 `.cpu()` 后的 `numpy`/Python 解包。
- Engine 层：只处理 CPU 结果，不保留 GPU tensor 引用。

## 审查清单

出现以下任意模式时，默认视为违反本 ADR，除非代码位于 runner 且符合四阶段顺序：

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

## 备选方案

### 方案 A：允许 engine 处理少量传输

不采纳。这样会让“调度层是否阻塞”变得不可推断，并持续扩大设备语义的扩散范围。

### 方案 B：允许 model 内部自行回传部分结果

不采纳。这样会破坏 runner 作为唯一设备边界的设计，并让同步点埋入深层模块。

## 后续执行项

1. 将 `server.encode_latents()` 的 GPU 搬运与执行迁移到 runner。
2. 为代码评审增加 Host/Device 边界检查项。
3. 在新增模型接入文档中要求：任何跨设备操作只能出现在 runner。
