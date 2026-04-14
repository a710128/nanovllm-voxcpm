# 推理传输与同步规范

本规范是 `docs/adr/0001-runner-owns-host-device-transfers.md` 的可执行版本。新增或修改推理代码时必须遵守。

## 1. 目标

推理热路径必须严格分为四个阶段。这里的“推理热路径”包含：逐 step 的生成路径，以及生成前的 prompt-audio latent encode 路径。

1. CPU 校验
2. Runner 内异步 H2D
3. GPU 纯计算且无同步
4. Runner 内统一 D2H

任何实现都不得改变这个顺序。

## 2. 适用范围

适用于以下目录中的推理/生成热路径：

- `nanovllm_voxcpm/engine/`
- `nanovllm_voxcpm/models/voxcpm/`
- 后续新增的 `nanovllm_voxcpm/models/*/`

## 3. 分层要求

### `server`

允许：

- 请求参数校验
- I/O 解析
- CPU 数据解码和归一化
- 把 CPU payload 传给 engine

禁止：

- 任何 `cuda` / `cpu` / `to(device)` 调用
- 任何读取 GPU tensor 值的操作

### `engine` / `scheduler`

允许：

- 调度
- KV 规划
- `Sequence` 状态推进
- 纯 CPU 后处理

禁止：

- 创建 GPU tensor
- 从 GPU 取回结果
- 依赖 GPU 中间态做校验或停止判断

### `model runner`

允许：

- 所有 H2D / D2H
- attention context 准备
- GPU forward / graph replay / decode
- 最终结果回传 CPU

约束：

- H2D 必须先于 GPU 计算全部完成准备
- GPU 计算期间不得触发同步
- D2H 必须晚于 GPU 计算完成
- pinned memory 与 `non_blocking=True` 是强烈建议，不是替代边界约束的独立目标

### `model` / `layers`

允许：

- 纯 device 计算

禁止：

- 直接访问 Python 标量化结果
- `.cpu()` / `.numpy()` / `.tolist()` / `.item()`
- 隐式或显式 Host/Device 传输

## 4. 禁止模式

以下模式在 runner 外一律禁止：

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

以下模式即使位于 runner，也只能出现在阶段 D：

```python
tensor.cpu()
tensor.numpy()
tensor.tolist()
tensor.item()
```

多个连续 D2H 读取是允许的，但它们必须发生在 GPU 计算结束之后，且不得与 GPU 计算交叉。

## 5. 推荐模式

### 阶段 A：CPU 校验

```python
if max_generate_length < 1:
    raise ValueError("max_generate_length must be >= 1")

payload = np.asarray(data, dtype=np.float32)
```

### 阶段 B：Runner 内异步 H2D

```python
gpu_inputs = torch.from_numpy(cpu_array).cuda(non_blocking=True)
gpu_scalar = torch.tensor(values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
```

### 阶段 C：GPU 纯计算

```python
outputs = self.run_model(inputs, is_prefill)
latents = outputs["latents"]
decoded = self.vae.decode(latents.permute(0, 2, 1))
```

### 阶段 D：Runner 内统一 D2H

```python
latents_cpu = latents.to(torch.float32).cpu().numpy()
stop_flags_cpu = outputs["stop_flag"].cpu().tolist()
```

## 6. PR / Code Review 检查项

- 是否把所有输入校验留在 CPU？
- 是否只有 runner 执行 H2D / D2H？
- GPU 计算期间是否完全没有同步？
- `postprocess_seq()` 是否只接收 CPU 数据？
- 是否新增了 runner 外的 `.cuda()`、`.cpu()`、`.item()`、`.tolist()`？

## 7. 对新增模型接入的要求

新增 `models/<family>/runner.py` 时，必须显式满足：

- 输入 payload 为 CPU 数据结构。
- `run()` 负责全部跨设备边界。
- 返回给 engine 的结果为 CPU 数据结构。
- 不把设备语义泄漏到 `engine.py`、`server.py`、`model.py`。

## 8. 现有代码基线

符合规范的典型位置：

- `nanovllm_voxcpm/engine/model_runner.py:292`
- `nanovllm_voxcpm/engine/model_runner.py:297`
- `nanovllm_voxcpm/models/voxcpm/runner.py:102`

需要整改的典型位置：

- `nanovllm_voxcpm/models/voxcpm/server.py:108`
