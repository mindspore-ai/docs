# 其它训练特性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/other_training_features.md)

除了优化器、学习率等训练超参数和并行策略，动态图（PyNative）训练还有几项常用特性：**梯度累积、梯度裁剪、融合算子与混合精度**。本页依次介绍每项特性的用途与配置方式，并给出可直接套用的 YAML 片段。

其余两类配置可参考对应文档：训练超参数（优化器、学习率与 fp32 主权重）见[训练超参数与优化器](./training_hyperparameters.md)，并行策略见[分布式并行训练](./parallel_training.md)；其中梯度累积与并行度、批大小直接相关，建议与并行文档配合阅读。完整字段含义见[配置文件说明](./configuration.md)。

## 特性速查

| 特性    | 配置入口                       | 关键字段                                                         | 默认行为                             |
|-------|----------------------------|--------------------------------------------------------------|----------------------------------|
| 梯度累积  | `training` / `parallelism` | `global_batch_size`、`local_batch_size`、`data_parallel_shard` | 由三者推导累积步数，不直接配步数                 |
| 梯度裁剪  | `training`                 | `max_norm`（默认 `1.0`）                                         | 默认开启，按全局 L2 范数裁剪                 |
| 融合算子  | `model`                    | `normalization`、`fused_norm`、`hidden_act`                    | 默认 `RMSNorm` + `fused_norm=True` |
| 混合精度  | `model`                    | `params_dtype`、`compute_dtype` 等                             | fp32 主权重 + bf16 计算               |

## 一、梯度累积

梯度累积是将一个全局 batch 拆分成若干 micro-batch，依次进行前向计算和反向传播，将梯度累积到一定步数后再统一更新优化器参数的技术，使显存受限时仍能模拟较大等效 batch 的训练效果。

### 适用场景

显存不足以一次性容纳目标全局批大小时，但又希望保持较大的等效 batch（提升收敛稳定性）时使用。

### 推导规则

动态图**不单独配置梯度累积步数**，而是由 `training` 段的 `global_batch_size`、`local_batch_size` 与数据并行维度 `data_parallel` 推导（见 [mindformers/pynative/trainer/trainer.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/trainer.py) 的 `_compute_data_parallel_size`）：

```text
num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)
```

其中 `data_parallel = world_size // (tensor_parallel * pipeline_parallel * context_parallel) = dp_replicate * dp_shard`，是完整的数据并行度（`dp_replicate` 为副本度，对应运行时属性 `data_parallel_replicate`；`dp_shard` 为分片度，即解析后的 `data_parallel_shard`）。用户只需配置分片度 `data_parallel_shard`；纯 FSDP（副本度为 1）时 `data_parallel` 即等于 `data_parallel_shard`。

**整除与默认推导**：

- **向下取整**：源码用整除 `//`。若 `global_batch_size` 不能被 `data_parallel * local_batch_size` 整除，多出的样本会被丢弃。建议让 `global_batch_size` 能被 `data_parallel * local_batch_size` 整除。
- **`data_parallel_shard` 默认值是 `-1`**。当其 `< 0` 时框架按 `dp_replicate = 1` 推导，此时 `data_parallel_shard` 被回写为 `data_parallel`（即分片度等于数据并行维度），不必手动填写；只有需要 HSDP（同时复制 + 分片）时才显式设置。

### 配置示例

下例的目标是让累积步数 `num_accumulation_steps = 4`。以单卡（`data_parallel = data_parallel_shard = 1`）、`local_batch_size = 2` 为例，需要 `global_batch_size = 8`：`8 // (1 * 2) = 4`。

```yaml
training:
  steps: 1000
  local_batch_size: 2        # 单卡每次前向的样本数（一个 micro-batch）
  global_batch_size: 8       # 每步全局样本数；8 // (1 * 2) = 4 个累积步
  max_norm: 1.0
  seed: 42

parallelism:
  data_parallel_shard: -1    # 默认 -1，单卡下推导为 data_parallel = 1
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1

optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01
```

### 字段说明

| 参数名称                   | 数据类型 | 是否可选 | 默认值   | 取值说明                                                    |
|------------------------|------|------|-------|---------------------------------------------------------|
| `global_batch_size`    | int  | 可选   | `1`   | 每步处理的全局样本数（配置段：`training`）。                             |
| `local_batch_size`     | int  | 可选   | `1`   | 单卡每个 micro-batch 的样本数（配置段：`training`）。                  |
| `data_parallel_shard`  | int  | 可选   | `-1`  | 数据并行分片度（配置段：`parallelism`）；`-1` 时按 `data_parallel` 推导。  |

> - **约束**：`data_parallel * local_batch_size` 不得超过 `global_batch_size`，否则会抛 `ValueError`，提示需增大 `global_batch_size` 或减小 `local_batch_size`。
> - **loss 自动缩放**：当 `num_accumulation_steps > 1` 时，框架对每个 micro-batch 的 loss 乘以 `1 / num_accumulation_steps` 再反向（见 `_forward_backward`），从而使累积后的梯度等价于一次大 batch 的平均梯度。各 micro-batch 的缩放后 loss 会在 `training_step` 中累加，因此上报的 `device_loss` 即为该全局 batch 的平均 loss（分布式下再跨卡 all-reduce 取平均）。

## 二、梯度裁剪

梯度裁剪是在优化器更新前对所有参数梯度的全局 L2 范数设置上限，当范数超过阈值时按比例缩放梯度，以抑制单步更新过大、稳定训练的一种技术。

### 适用场景

训练中出现 loss 抖动、梯度范数尖刺时使用。动态图默认即开启梯度裁剪。

### 工作机制

框架在优化器更新前，先计算所有参数梯度的全局 L2 范数（分布式下跨进程 all-reduce 同步），当范数超过 `max_norm` 时按比例就地缩放梯度（见 [mindformers/pynative/trainer/utils.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/utils.py) 的 `_calculate_global_grad_norm`）。

**数值与分布式一致性**：范数计算在 **fp32** 精度下进行，以避免 bf16/fp16 下的数值不稳定；并对 DTensor 的 `Replicate` placement 维度做缩放校正，保证多卡结果与单卡一致。

### 配置示例

大模型训练初期 loss 不稳，可以收紧裁剪阈值，抑制异常步：

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 8
  seed: 42
  max_norm: 0.5      # 收紧裁剪上限
```

### 字段说明

| 参数名称        | 数据类型   | 是否可选 | 默认值    | 取值说明                                                                                                   |
|-------------|--------|------|--------|--------------------------------------------------------------------------------------------------------|
| `max_norm`  | float  | 可选   | `1.0`  | 梯度全局 L2 范数上限（配置段：`training`）。应保持正数，不要设为 `0` 或负值——设为 `0` 会导致梯度被整体缩放到零（等效于不更新），设为负值则会使梯度反号。需减弱裁剪时请调大该值。  |

## 三、融合算子

融合算子将归一化、激活等相邻的多个算子合并为单一 kernel 执行，减少动态图模式下的 kernel 启动开销，从而提升训练吞吐量。

### 适用场景

动态图层默认已启用融合算子，一般无需改动；当需要切换归一化类型或激活函数时，在 `model` 段配置即可。

### 算子一览

源码位于 [mindformers/pynative/layers/layer_norm.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/layers/layer_norm.py) 与 [mindformers/pynative/layers/activation.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/layers/activation.py)。

| 算子        | 类                | 关键参数（默认）                           | 行为                                                                      |
|-----------|------------------|------------------------------------|-------------------------------------------------------------------------|
| RMSNorm   | `FusedRMSNorm`   | `eps=1e-5`、`compute_dtype=float32` | 基于 MindSpore 融合 `rms_norm`；先 cast 到 `compute_dtype` 计算，再 cast 回输入 dtype |
| LayerNorm | `FusedLayerNorm` | `eps=1e-5`、`compute_dtype=float32` | 基于融合 `layer_norm`；同样按 `compute_dtype` 计算后 cast 回原 dtype                 |
| GELU      | `GELU`           | `approximate="none"`               | 支持 `approximate` 取 `none` / `tanh`                                      |
| SiLU      | `SiLU`           | `inplace=False`                    | 支持 `inplace` 原地更新                                                       |
| SwiGLU    | `FusedSwiGlu`    | —                                  | 基于 `swiglu` 算子的融合 SwiGLU                                                |

**归一化与激活的可选值**：

- **归一化**：由 `normalization` 字段选择，默认 `"RMSNorm"`，**仅支持 `"LayerNorm"` 与 `"RMSNorm"` 两个取值**（由 `get_norm_cls` 解析）。
- **融合开关**：由 `fused_norm` 字段控制，默认 `True`。当前 `get_norm_cls` **只支持融合实现**，`fused_norm=False` 会直接报错，因此保持默认即可。
- **激活**：由 `hidden_act` 字段选择，`ACTIVATION_MAP` 仅注册 `gelu`、`silu`、`fusedswiglu` 三类。

### 配置示例

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  hidden_size: 1792
  num_hidden_layers: 12
  num_attention_heads: 8
  # 融合算子相关
  normalization: "RMSNorm"   # 可选 "RMSNorm" / "LayerNorm"，默认 "RMSNorm"
  fused_norm: True           # 默认 True，保持融合实现
  rms_norm_eps: 1.e-6        # RMSNorm 的 eps
  hidden_act: "silu"         # 可选 "gelu" / "silu" / "fusedswiglu"
```

**eps 字段**：`FusedRMSNorm` / `FusedLayerNorm` 构造参数 `eps` 默认 `1e-5`。模型侧通常通过 `rms_norm_eps` 等模型字段传入实际值（如 DeepSeek-V3 用 `1.e-6`），以 `model` 段的取值为准。

## 四、混合精度

混合精度训练是指在训练过程中对不同计算层次使用不同的数值精度——主体计算（如矩阵乘）使用 bf16/fp16 等低精度以提升吞吐，而对归一化、softmax、RoPE 等数值敏感算子保留 fp32 以保证数值稳定性。

### 适用场景

在 Ascend 上以 bf16 进行主体计算可显著提升吞吐，同时对归一化、softmax、RoPE 等数值敏感算子保留 fp32，兼顾速度与稳定性。混合精度通过 `model` 段的一组 dtype 字段控制。

**model 段是可扩展的**：`model` 段 `allow_extra = True`，字段随模型扩展，**不同模型可用的 dtype 字段名可能不同**。例如 `router_dense_type` 仅 MoE 模型有意义。这些 dtype 字段在代码中没有内置默认值，实际取值来自模型 YAML；下表与示例的取值以 DeepSeek-V3 配置 `pynative_ds3.yaml` 为准，落地时请对照所用模型的实际字段。

### 字段说明

| 参数名称                      | 数据类型 | 是否可选 | 默认值          | 取值说明                            |
|---------------------------|------|------|--------------|---------------------------------|
| `params_dtype`            | str  | 可选   | `"float32"`  | 参数（主权重）保存精度。                    |
| `compute_dtype`           | str  | 可选   | `"bfloat16"` | 矩阵乘等主体计算精度，通常用 bf16 提升吞吐。       |
| `layernorm_compute_dtype` | str  | 可选   | `"float32"`  | 归一化计算精度，保留 fp32 保证数值稳定。         |
| `softmax_compute_dtype`   | str  | 可选   | `"float32"`  | softmax 计算精度。                   |
| `rotary_dtype`            | str  | 可选   | `"float32"`  | 旋转位置编码（RoPE）计算精度。               |
| `router_dense_type`       | str  | 可选   | `"float32"`  | MoE router 计算精度（**仅 MoE 模型**）。  |

**fp32 主权重副本**：当 `compute_dtype` 为 bf16/fp16 时，优化器会为这些参数额外维护一份 **fp32 主权重副本**：优化器更新在 fp32 上完成后再同步回低精度参数，避免低精度直接累加权重的精度损失。详见[训练超参数与优化器](./training_hyperparameters.md)。

### 场景化配置

**场景 A：DeepSeek-V3（MoE 模型）**

含 MoE，需配置 `router_dense_type`；敏感算子全部保留 fp32：

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  params_dtype: "float32"            # 主权重 fp32
  compute_dtype: "bfloat16"          # 主体计算 bf16
  layernorm_compute_dtype: "float32" # 归一化 fp32
  softmax_compute_dtype: "float32"   # softmax fp32
  rotary_dtype: "float32"            # RoPE fp32
  router_dense_type: "float32"       # MoE router fp32（仅 MoE）
```

**场景 B：通用稠密模型**

非 MoE 模型无需 `router_dense_type`，其余字段保持稳态混合精度配置：

```yaml
model:
  params_dtype: "float32"            # 主权重 fp32
  compute_dtype: "bfloat16"          # 主体计算 bf16
  layernorm_compute_dtype: "float32"
  softmax_compute_dtype: "float32"
  rotary_dtype: "float32"
```

**选型建议**：

- **稳态优先（推荐）**：`params_dtype=float32` + `compute_dtype=bfloat16`，配合 fp32 主权重与 fp32 归一化/softmax/RoPE，精度稳定，是默认方案。
- **吞吐优先**：可将更多敏感算子也切到 bf16，但需密切关注 loss 与梯度范数；一旦出现发散，优先把 `layernorm_compute_dtype` / `softmax_compute_dtype` 调回 `float32`。

## 相关文档

- 优化器、学习率与 fp32 主权重：[训练超参数与优化器](./training_hyperparameters.md)
- 并行维度（DP/TP/CP/PP）与批大小推导：[分布式并行训练](./parallel_training.md)
- 数据集配置与每卡批大小：[数据集](./dataset.md)
- 配置文件字段总览：[配置文件说明](./configuration.md)
- 训练全流程：[训练指南](../guide/training.md)
- 快速上手：[快速开始](../quick_start/quick_start.md)
