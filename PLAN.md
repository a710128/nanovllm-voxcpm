# VoxCPM2 独立后端接入计划

## 目标
- 在 `nanovllm_voxcpm/models/voxcpm2/` 下新增一套**完全独立**的 VoxCPM2 推理后端。
- 不与现有 `nanovllm_voxcpm/models/voxcpm/` 目录产生代码交集，避免对 VoxCPM1 造成回归影响。
- 第一阶段优先打通基础推理链路与服务化接入；不要求补齐上游 `prompt_wav_path + librosa + VAD trim` 的完全一致行为。

## 现状结论

### 1. 当前 VoxCPM1 在 nano-vllm 中的适配方式
- `nanovllm_voxcpm/llm.py` 根据 `config.json` 中的 `architecture` 分发模型后端；当前仅支持 `voxcpm`。
- `nanovllm_voxcpm/models/voxcpm/` 下形成完整推理链路：
  - `server.py`：对外服务接口、进程池、LoRA 管理
  - `engine.py`：请求组装、sequence 状态管理、prompt latents 拼接
  - `runner.py`：batch flatten、调用模型、VAE 解码
  - `model.py`：适配 nano-vllm KV cache / CUDA graph / tensor-parallel 的推理模型实现
- 当前接入不是直接复用上游 `VoxCPM/src/voxcpm/model/voxcpm.py`，而是按 nano-vllm 的执行框架重写了推理版本。

### 2. VoxCPM2 相对 VoxCPM1 的主要变化
- 配置默认值变化：
  - `patch_size: 2 -> 4`
  - `residual_lm_num_layers: 6 -> 8`
  - `scalar_quantization_latent_dim: 256 -> 512`
  - `max_length: 4096 -> 8192`
- 音频 VAE 从 `AudioVAE` 升级为 `AudioVAEV2`，新增：
  - `out_sample_rate`
  - `sr_bin_boundaries`
  - `cond_type`
  - `cond_dim`
  - `cond_out_layer`
- Local DiT 从 `VoxCPMLocDiT` 升级为 `VoxCPMLocDiTV2`。
- residual 分支融合方式变化：
  - VoxCPM1：`enc_outputs + masked_feat_embed`
  - VoxCPM2：`fusion_concat_proj(cat(enc_outputs, masked_feat_embed))`
- DiT 条件隐藏状态融合方式变化：
  - VoxCPM1：`lm_to_dit_proj(...) + res_to_dit_proj(...)`
  - VoxCPM2：`cat(lm_to_dit_proj(...), res_to_dit_proj(...))`
- 上游 VoxCPM2 在纯 PyTorch 入口中新增 `librosa` + VAD trim 的 prompt wav 预处理，但该逻辑不属于 nano-vllm 第一阶段必须项。

## 接入原则
- 新增 `nanovllm_voxcpm/models/voxcpm2/` 独立目录。
- 不复用、不引用 `nanovllm_voxcpm/models/voxcpm/` 目录内实现。
- 与现有 VoxCPM1 仅在**框架公共层**共享：
  - `nanovllm_voxcpm/config.py`
  - `nanovllm_voxcpm/engine/*`
  - `nanovllm_voxcpm/layers/*`
  - `nanovllm_voxcpm/utils/*`
- 对外服务接口尽量与当前 VoxCPM1 保持一致，降低 deployment 层改动。

## 实施方案

### 阶段 1：独立后端骨架
新增目录：`nanovllm_voxcpm/models/voxcpm2/`

计划文件：
- `__init__.py`
- `config.py`
- `model.py`
- `runner.py`
- `engine.py`
- `server.py`
- `utils.py`（如需要，独立维护 tokenizer / 辅助逻辑）

目标：
- 建立一套与 `voxcpm/` 平行的完整后端。
- 保持 runner-model 的接口契约稳定：
  - 输入继续走 `positions / text_tokens / feat / feat_mask / temperature / cfg_value`
  - 输出继续返回 `latents / stop_flag`

### 阶段 2：配置与分发
- 在 `nanovllm_voxcpm/llm.py` 中新增 `architecture == "voxcpm2"` 分支。
- `config.json` 解析走 `nanovllm_voxcpm/models/voxcpm2/config.py`。
- `max_model_len` 默认策略按 VoxCPM2 能力评估，优先支持到 8192。

### 阶段 3：模型主干适配
在 `nanovllm_voxcpm/models/voxcpm2/model.py` 中实现独立的 `VoxCPM2Model`：

- 保留 nano-vllm 推理模型风格：
  - 支持 prefill / decode 共用单一 forward
  - 兼容 KV cache / flash attention / CUDA graph
- 引入 VoxCPM2 特有结构：
  - `fusion_concat_proj`
  - `VoxCPMLocDiTV2`
  - V2 版本 residual 输入融合逻辑
  - V2 版本 DiT hidden 拼接逻辑
- 保持 packed modules / fused linear 命名与权重加载逻辑兼容；如 VoxCPM2 层命名不同，则同步调整 loader 映射。

### 阶段 4：AudioVAEV2 适配
- 在公共层新增独立 V2 实现，建议新增文件：
  - `nanovllm_voxcpm/layers/audio_vae_v2.py`
- 不修改现有 `nanovllm_voxcpm/layers/audio_vae.py` 的 V1 行为。
- 实现内容包括：
  - `AudioVAEV2`
  - `AudioVAEConfigV2`
  - sample-rate conditioned decoder
  - `decode(z, sr_cond=None)`
  - `encode(audio_data, sample_rate)`
- `runner.py` 中按 VoxCPM2 config 实例化并加载 `audiovae.pth`。

### 阶段 5：Engine / Runner / Server 适配

#### engine.py
- 复刻并独立维护 VoxCPM2 的 sequence payload。
- 继续支持：
  - 无 prompt 生成
  - `prompt_text + prompt_latents` 生成
- 保留 `max_model_len` 检查。

#### runner.py
- 负责：
  - flatten batch inputs
  - 调用 `VoxCPM2Model`
  - 用 `AudioVAEV2` 解码 waveform
- 维持与当前框架兼容的输出结构。

#### server.py
- 新增 `AsyncVoxCPM2ServerPool` / `SyncVoxCPM2ServerPool`。
- 保持以下对外接口尽量一致：
  - `generate`
  - `encode_latents`
  - `get_model_info`
  - `load_lora`
  - `set_lora_enabled`
  - `reset_lora`

### 阶段 6：LoRA 兼容性
- 第一阶段目标：优先保证 base model 跑通。
- 若 VoxCPM2 与 VoxCPM1 的 fused projection 命名一致，则复用当前公共 loader 规则。
- 若 `VoxCPMLocDiTV2` 或新增投影层命名不同：
  - 扩展 `nanovllm_voxcpm/utils/loader.py`
  - 必要时为 VoxCPM2 增加专属 LoRA name mapping
- `fusion_concat_proj` 是否纳入 LoRA 目标，作为第二优先级决策项。

## 第一阶段明确不做的内容
- 不要求与上游 `voxcpm2.py` 的 `prompt_wav_path` 路径完全对齐。
- 不要求第一版引入 `librosa` 和 `_trim_audio_silence_vad` 到服务主链路。
- 不要求第一版完成所有 LoRA 权重映射验证。
- 不要求第一版与 VoxCPM1 共用 `models/voxcpm/` 目录内任何代码。

## 验证方案

### A. 单元测试
- 为 `nanovllm_voxcpm/llm.py` 新增 `architecture == "voxcpm2"` 分发测试。
- 为 `nanovllm_voxcpm/models/voxcpm2/engine.py` 增加：
  - prompt 长度超限校验
  - `prompt_len + max_generate_length` 超限校验
  - 边界长度可接受校验
- 为 `AudioVAEV2` 的 config 构造与 encode/decode 基本路径增加轻量测试。

### B. 语法与静态可执行性
- `uv run python -m compileall nanovllm_voxcpm deployment tests`

### C. 冒烟验证
在具备权重与 GPU 环境时验证：
- 能成功加载 VoxCPM2 模型目录
- server 能正常启动
- `encode_latents` 可返回合法 latent bytes
- 无 prompt 输入时可生成短音频
- `prompt_text + prompt_latents` 输入时可生成短音频
- 输出无 NaN / 无 shape mismatch / stop 行为正常

### D. 对齐验证
在条件允许时，与上游 `VoxCPM2Model.from_local(...).generate(...)` 做功能对比：
- 长度范围是否合理
- 是否能正常停止
- 是否存在明显异常输出

## 风险点
- `AudioVAEV2` 是最大改动点，不能直接套用现有 V1 AudioVAE。
- `VoxCPMLocDiTV2` 需要确认在 nano-vllm 当前 non-causal attention 实现下可平滑适配。
- VoxCPM2 的 `max_length` 提升到 8192 后，CUDA graph / block table / 显存占用需要重新评估。
- LoRA 若涉及新增投影层，name mapping 可能需要额外设计。

## 交付顺序
1. 建立 `voxcpm2/` 独立后端骨架
2. 新增 `llm.py` 分发与基础 config
3. 适配 `VoxCPM2Model`
4. 新增 `AudioVAEV2`
5. 打通 runner / engine / server
6. 增加单测与 compileall 验证
7. 有 GPU 和权重时做真实冒烟验证

## 完成标准
- 仓库内存在独立的 `nanovllm_voxcpm/models/voxcpm2/` 后端。
- `VoxCPM.from_pretrained(...)` 能正确识别并加载 `architecture == "voxcpm2"`。
- VoxCPM1 与 VoxCPM2 后端代码相互隔离。
- VoxCPM2 至少能完成 base model 的服务化推理闭环。
- 单测与基础语法校验通过。
