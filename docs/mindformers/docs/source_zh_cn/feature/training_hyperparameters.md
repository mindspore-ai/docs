# 训练超参数与优化器

动态图（PyNative）训练的超参数集中在配置文件的三个**顶层并列段**中：`optimizer`（优化器）、`lr_scheduler`（学习率策略）与 `training`（训练基础参数）。三者分别由 `OptimizerConfig`、`LrSchedulerConfig`、`TrainingConfig` 解析（见 [配置文件说明](./configuration.md)）。

本页详细介绍了优化器，学习率策略，训练基础参数的配置和示例，并给出可直接套用的配套组合。整体训练流程可先参考 [概述](../introduction/overview.md) 与 [快速开始](../quick_start/quick_start.md)。

启动训练时，配置文件作为 `--config` 传入，动态图须显式指定 `--mode 1`：

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py --config xxx.yaml --mode 1" ...
```

---

## 选型速查

**优化器选型**：动态图当前支持 `AdamW` 与 `Muon` 两种。

| 维度 | AdamW | Muon |
|---|---|---|
| 适用模型 | 通用，无限制 | **仅** 启用 Multi-Latent Attention 的模型（如 DeepSeek-V3） |
| 是否支持 SWAP | 支持 | **不支持** |
| 多卡通信 | 无需额外通信策略 | 二维分片权重需 all-gather / P2P 聚合（`comm_strategy`） |
| 默认 `learning_rate` | `1e-5`（由 `lr_scheduler` 提供） | `1e-5`（由 `lr_scheduler` 提供） |
| 典型场景 | 默认首选；预训练、微调通用 | DeepSeek-V3 类 MLA 大模型预训练，追求收敛质量 |

> AdamW 是通用默认优化器，适配所有模型与并行策略。Muon 仅适用于 MLA 模型（如 DeepSeek-V3），且不能与 SWAP 同时开启，否则会在优化器构造阶段直接报错。

**学习率调度器选型**：所有带 warm-up 的调度器都先线性升温到基础学习率，warm-up 之后的行为各不相同。

| 调度器 `type` | warm-up 后行为 | 典型用途 | 关键扩展字段 |
|---|---|---|---|
| `ConstantWarmUpLR` | 保持常数 | 调试、续训对齐、短任务 | — |
| `LinearWithWarmUpLR` | 线性衰减到 0 | 简单微调 | — |
| `CosineWithWarmUpLR` | 余弦衰减 | **预训练最常用** | `num_cycles`、`lr_end`、`decay_steps` |
| `CosineWithRestartsAndWarmUpLR` | 带重启的余弦 | 长训练周期性重启 | `num_cycles`（重启次数）、`decay_steps` |
| `PolynomialWithWarmUpLR` | 多项式衰减 | 需自定义衰减曲线 | `power`、`lr_end`、`decay_steps` |
| `WarmUpStableDecayLR` | 升温→恒定→衰减三段（WSD） | 大规模预训练，便于中途扩 token | `lr_end`、`decay_start_steps`/`decay_start_ratio` |
| `CosineAnnealingLR` | 余弦退火（无 warm-up） | 简单周期退火 | `t_max`、`eta_min`（注意基础学习率字段名为 `base_lr`） |

> 学习率调度器复用 `mindformers/core/lr` 的 LR 注册表，上表之外还注册了 `ConstantWithCoolDownLR`、`CosineAnnealingWarmRestarts`、`LearningRateWiseLayer` 等变体，按需启用（注册名以 `mindformers/core/lr/lr_schedule.py` 的 `__all__` 为准）。

---

## 一、优化器（optimizer）

`AdamW` 与 `Muon` 两种优化器都**始终为 bf16/fp16 参数创建 fp32 主权重副本**（`_init_main_params`）：优化器的动量/方差状态与参数更新均在 fp32 精度上完成，再写回低精度模型参数，与 Megatron 的混合精度优化器设计对齐。

`optimizer` 段标注 `allow_extra = True`，允许传入字段表以外的额外参数（如 `use_fused`）。

### 1.1 AdamW

**适用场景**：通用默认优化器，对所有模型、所有并行策略均适用。预训练、微调若无特殊诉求，优先使用 `AdamW`。

```yaml
optimizer:
  type: AdamW
  betas:
    - 0.9
    - 0.95
  eps: 1.e-8
  weight_decay: 0.01
  accumulate_allreduce_grads_in_fp32: True
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | str | `AdamW` | 优化器类型。 |
| `betas` | list[float] | `[0.9, 0.95]` | 一阶/二阶矩的指数衰减率，长度须为 2，每项取值 `[0, 1)`。 |
| `eps` | float | `1.0e-8` | 分母数值稳定项，须 `> 0`。 |
| `weight_decay` | float | `0.01` | 解耦的权重衰减（AdamW 风格 L2 正则），须 `>= 0`。 |
| `weight_decay_include` | list[str] | `None` | 强制施加权重衰减的参数名匹配规则。 |
| `weight_decay_exclude` | list[str] | `None` | 强制不施加权重衰减的参数名匹配规则。 |
| `accumulate_allreduce_grads_in_fp32` | bool | `True` |「梯度累积 + 优化器更新」整条链路是否都在 fp32 上进行。 |

**关键字段怎么调**：

- `betas`：第一项控制一阶动量平滑、第二项控制二阶矩平滑。预训练大模型常用 `[0.9, 0.95]`；小数据微调可设置为 `[0.9, 0.999]`。
- `eps`：仅作数值兜底，一般无需改动；bf16 训练若出现除零类异常可适当增大（如 `1e-6`）。
- `weight_decay`：典型预训练取 `0.01~0.1`。配合下面的 `weight_decay_include` / `weight_decay_exclude` 控制作用范围。
- **`weight_decay_include` / `weight_decay_exclude`**：两者都是参数名匹配规则列表，用于**覆盖**默认的衰减归属——`include` 命中的参数被强制施加权重衰减，`exclude` 命中的参数被强制排除。常见做法是把 LayerNorm、bias、embedding 等放入 `exclude`，仅对线性层权重做衰减：

  ```yaml
  optimizer:
    type: AdamW
    weight_decay: 0.1
    weight_decay_exclude:
      - "*norm*"
      - "*bias*"
  ```

- **accumulate_allreduce_grads_in_fp32**：默认 `True`。开启后，框架为 bf16/fp16 参数注册反向 hook，把回传梯度**先转成 fp32 再累积**，于是「梯度累积 + 优化器更新」整条链路都在 fp32 上进行，与始终存在的 fp32 主权重副本配套，能显著减少低精度下的累积误差。除非显存极度紧张，建议保持开启。

### 1.2 Muon

Muon 对二维/三维权重做 Newton-Schulz 迭代正交化，对 word embedding、输出层等其余权重回退到内置 AdamW。它在 MLA 大模型上往往带来更好的收敛质量，但**有严格的使用前提**。

Muon 在 `__init__` 中会调用 `_verify_model` 并检查 SWAP 配置，以下任一条件不满足都会在**优化器构造阶段**抛出 `ValueError`，训练无法启动：

1. **必须传入 `model`**：否则报 `Model must be provided for Muon optimizer.`（由框架在构建优化器时自动注入，用户无需在 YAML 手写）。
2. **不支持 Legacy Model**：模型为 Legacy 实现时报 `Muon does not support Legacy Model.`。
3. **必须启用 Multi-Latent Attention**：模型配置 `multi_latent_attention` 必须为 `True`，否则报 `... only supports models with Multi-Latent Attention enabled.`。
4. **不支持 SWAP**：传入 `swap=True` 时报 `Muon does not support swap.`。

```yaml
optimizer:
  type: Muon
  weight_decay: 0.1
  momentum: 0.95
  matched_adamw_rms: 0.2
  nesterov: True
  eps: 1.e-7
  ns_steps: 5
  ns_coefficients: [3.4445, -4.7750, 2.0315]
  adamw_betas:
    - 0.95
    - 0.95
  adamw_eps: 1.e-8
  qk_clip_enabled: True
  qk_clip_threshold: 100
  comm_strategy: allgather
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `weight_decay` | float | `0.1` | 权重衰减。 |
| `momentum` | float | `0.95` | Muon 动量系数。 |
| `matched_adamw_rms` | float | `0.2` | 与 AdamW 更新幅度对齐的 RMS 缩放系数。 |
| `nesterov` | bool | `True` | 是否使用 Nesterov 动量。 |
| `eps` | float | `1.0e-7` | Newton-Schulz 归一化的数值稳定项。 |
| `ns_steps` | int | `5` | Newton-Schulz 迭代步数（仅扁平 `ns_coefficients` 形式生效）。 |
| `ns_coefficients` | tuple/list | `(3.4445, -4.7750, 2.0315)` | Newton-Schulz 系数，支持扁平三元组或分段调度。 |
| `adamw_betas` | tuple | `(0.95, 0.95)` | 回退到 AdamW 的权重所用 betas。 |
| `adamw_eps` | float | `1.0e-8` | 回退 AdamW 的 eps。 |
| `qk_clip_enabled` | bool | `True` | 是否对注意力 logit 施加 QK-Clip 缩放。 |
| `qk_clip_threshold` | float | `100` | QK-Clip 阈值，须 `> 0`（`qk_clip_enabled=True` 时校验）。 |
| `comm_strategy` | str | `allgather` | 多卡通信策略，详细描述可见[comm_strategy 的取舍](#comm_strategy-的取舍)章节。 |
| `use_fused_adamw` | bool | `False` | 非 Muon 权重是否使用融合 AdamW 算子。 |
| `adamw_include` | list[str] | `None` | 指定哪些参数使用 AdamW 而非 Muon；默认 `["*word_embeddings*", "*output_layer*"]`（2D/3D 权重使用 Muon，embedding/输出层使用 AdamW）。 |

#### ns_coefficients 的两种写法

`ns_coefficients` 经 `_normalize_ns_schedule` 归一化为「每步一个三元组」的调度，支持两种 YAML 写法：

- **扁平三元组** `[a, b, c]`（默认）：同一组系数应用于全部 `ns_steps` 次迭代。
- **分段调度** `[[[a, b, c], count], ...]`：每段把自身三元组重复 `count` 次，**总迭代步数为各段 `count` 之和，此时 `ns_steps` 被忽略**。适合让前几步与后几步用不同系数（如 DeepSeek V4 的前 8 步 / 后 2 步方案）。

分段调度示例：

```yaml
optimizer:
  type: Muon
  ns_coefficients:
    - [[3.4445, -4.7750, 2.0315], 8]   # 前 8 步用这组系数
    - [[2.0, -1.5, 0.5], 2]            # 后 2 步切换系数；总步数 = 8 + 2 = 10
  # 注意：使用分段调度时，ns_steps 字段不再生效
```

#### comm_strategy 的取舍

| 取值 | 行为 | 适用场景与代价 |
|---|---|---|
| `allgather`（默认） | 每张卡都 all-gather 全量权重并各自独立运行 Newton-Schulz | 实现简单；多卡时存在**冗余的 NS 计算**（每卡重复算同一份） |
| `allgather_deredundency` | 二维分片权重 P2P 聚合到指定 rank，NS 仅在该 rank 上计算后再分发 | **多卡训练**且 Muon 二维权重较多时开启，去除冗余 NS 计算、降低 HCCS 流量；代价是引入 P2P 聚合/分发通信与 rank 间负载分配逻辑 |

> 单卡训练无通信，保持默认 `allgather` 即可；大规模多卡（尤其专家/张量并行下二维权重多）时再评估 `allgather_deredundency`。

---

## 二、学习率策略（lr_scheduler）

`lr_scheduler` 段用 `type` 选择调度器，`learning_rate` 为 **warm-up 之后的基础学习率**，其余字段随调度器类型扩展（该段 `allow_extra = True`）。

> 调度器的总步数 `total_steps` 由框架在 `_build_lr_scheduler` 中用 `training.steps` 自动填入。用户**不要**在 `lr_scheduler` 段手填 `total_steps`，否则可能与实际训练步数不一致。
>
> `warmup_ratio` 与 `warmup_steps` 用于确定 warm-up 步数。`warmup_steps` 为绝对步数；`warmup_ratio` 为占总步数的比例。框架按 `_get_lr_steps` 处理——**只设其中一个**：填了 `warmup_ratio` 则用 `ratio × total_steps` 计算，否则使用 `warmup_steps`。同时填写会以 `warmup_ratio` 优先并忽略 `warmup_steps`。

下面给出每种调度器的最小可用片段（字段名以 `mindformers/core/lr/lr_schedule.py` 各 `__init__` 签名为准）。

### 2.1 ConstantWarmUpLR — 升温后保持常数

```yaml
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_steps: 100        # 或用 warmup_ratio
  warmup_lr_init: 0.0
```

特有字段：仅 warm-up 相关（`warmup_steps` / `warmup_ratio` / `warmup_lr_init`），warm-up 后恒定。

### 2.2 LinearWithWarmUpLR — 升温后线性衰减

```yaml
lr_scheduler:
  type: LinearWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
```

线性衰减到 0；总步数取自框架注入的 `total_steps`，无额外曲线字段。

### 2.3 CosineWithWarmUpLR — 升温后余弦衰减（预训练最常用）

```yaml
lr_scheduler:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
  lr_end: 0.0              # 余弦衰减终点
  num_cycles: 0.5         # 半周期，从峰值衰减到 lr_end
  # decay_steps: 9000     # 可选：自定义衰减步数，不填则用 total_steps
```

特有扩展字段：`num_cycles`（默认 `0.5`）、`lr_end`（默认 `0.0`）、`decay_steps`/`decay_ratio`。

### 2.4 CosineWithRestartsAndWarmUpLR — 带重启的余弦

```yaml
lr_scheduler:
  type: CosineWithRestartsAndWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  lr_end: 0.0
  num_cycles: 1.0         # 重启次数（>=1）
```

特有扩展字段：`num_cycles`（默认 `1.0`，表示重启周期数）、`lr_end`、`decay_steps`。

### 2.5 PolynomialWithWarmUpLR — 升温后多项式衰减

```yaml
lr_scheduler:
  type: PolynomialWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  lr_end: 1.e-7           # 衰减终点
  power: 1.0              # 多项式幂次；1.0 即线性，>1 衰减更快
```

特有扩展字段：`power`（默认 `1.0`）、`lr_end`（默认 `1e-7`）、`decay_steps`。

### 2.6 WarmUpStableDecayLR — 升温/恒定/衰减三段式（WSD）

```yaml
lr_scheduler:
  type: WarmUpStableDecayLR
  learning_rate: 1.e-5
  warmup_steps: 100
  warmup_lr_init: 0.0
  lr_end: 1.e-7
  decay_start_ratio: 0.8  # 在总步数 80% 处开始衰减；或用 decay_start_steps 指定绝对步
```

特有扩展字段：`lr_end`（默认 `1e-7`）、`decay_start_steps`（绝对起始步）或 `decay_start_ratio`（占总步数比例，二选一）。WSD 的优势是「恒定段」可随时延长，方便在不改曲线形状的前提下追加训练 token。

### 2.7 CosineAnnealingLR — 余弦退火（无 warm-up）

```yaml
lr_scheduler:
  type: CosineAnnealingLR
  base_lr: 1.e-5          # 注意：本调度器用 base_lr，而非 learning_rate
  t_max: 1000             # 半个余弦周期的步数
  eta_min: 0.0            # 退火下限
```

特有扩展字段：`t_max`（半周期步数，必填正整数）、`eta_min`（最小学习率，默认 `0.0`）。

> 该调度器的基础学习率参数名是 `base_lr`（而非其它调度器的 `learning_rate`），且不含 warm-up 阶段。迁移配置时注意这一差异。

---

## 三、训练基础参数（training）

`training` 段控制训练步数、批大小、梯度裁剪与可复现性等全局行为。

```yaml
training:
  steps: 1000           # 总训练步数（同时作为 LR 调度器的 total_steps）
  local_batch_size: 2   # 单卡（per-rank）batch size
  global_batch_size: 4  # 每步全局处理的样本总数
  max_norm: 1.0         # 梯度裁剪阈值，> 0 时启用
  seed: 42              # 随机种子
  deterministic: False  # 确定性计算开关
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `steps` | int | `1000` | 总训练步数；同时作为学习率调度器的总步数。 |
| `local_batch_size` | int | `1` | 单卡 batch size，必须为正。 |
| `global_batch_size` | int | `1` | 每步全局处理的样本总数，必须为正。 |
| `max_norm` | float | `1.0` | 全局梯度裁剪阈值，须为正数（典型取 `1.0`）。 |
| `seed` | int | `42` | 随机种子。 |
| `deterministic` | bool | `False` | 是否启用确定性训练。 |

### 3.1 批大小与梯度累积步数

`global_batch_size`, `local_batch_size` 与 `data_parallel_shard` 三者共同决定**每步是否需要梯度累积**：

```text
梯度累积步数 = global_batch_size // (local_batch_size × data_parallel_shard)
```

即每张数据并行卡每步处理 `local_batch_size` 个样本，全局一次累积满 `global_batch_size` 个样本才做一次优化器更新。框架按 `local_batch_size × data_parallel_shard` 对 `global_batch_size` 做整数除法推导累积步数（非整除不会报错，而是向下取整，有效全局批相应变小）。这里的 `data_parallel_shard` 是 FSDP 切分数：纯 FSDP（默认 `data_parallel_shard: -1`）时等于数据并行度，开 HSDP时小于数据并行度。其推导含义与设置见 [分布式并行训练](./parallel_training.md)。

### 3.2 梯度裁剪 max_norm

每次优化器更新前都会计算所有参数梯度的全局 L2 范数（`_calculate_global_grad_norm`）：当全局范数超过 `max_norm` 时，按 `max_norm / 全局范数` 的比例就地缩放全部梯度，从而把范数压回阈值，抑制梯度尖峰、稳定训练。大模型预训练常取 `1.0`。

> max_norm 应保持正数，不要用 0 或负值关闭裁剪。动态图路径没有「关闭裁剪」的开关：裁剪逻辑在 `clip_coef = max_norm / (全局范数 + eps) < 1` 时触发。若把 `max_norm` 设为 `0`，`clip_coef` 恒小于 1，会把梯度整体缩放到 0，相当于不再更新；设为负值则会让梯度反号。需要更弱的裁剪请调大 `max_norm`，而不是设为 0 或负数。

### 3.3 可复现性 seed 与 deterministic

- `seed`：统一初始化与数据打乱等随机源，保证同配置可复现。
- `deterministic`：开启后强制使用确定性算子，使多次运行结果逐位一致，便于调试与精度对齐。

> deterministic 会降低性能。确定性计算会牺牲部分算子的并行优化，**显著降低训练吞吐**。仅在排查精度问题、做逐位复现实验时开启；常规训练保持 `deterministic: False`。

---

## 四、组合示例（完整 YAML）

`optimizer`、`lr_scheduler`、`training` 是 YAML 的顶层并列键，与 `parallelism`、`train_dataset`、`model` 等同级。以下给出两套可直接套用的成段配置（已省略数据集/模型等无关段，实际使用请补全）。

### 4.1 AdamW 通用预训练

适用于绝大多数模型。

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 4
  max_norm: 1.0
  seed: 42
  deterministic: False

optimizer:
  type: AdamW
  betas:
    - 0.9
    - 0.95
  eps: 1.e-8
  weight_decay: 0.01
  accumulate_allreduce_grads_in_fp32: True
  weight_decay_exclude:
    - "*norm*"
    - "*bias*"

lr_scheduler:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
  lr_end: 0.0
```

### 4.2 Muon + DeepSeek-V3（MLA）

Muon 仅适用于启用 MLA 的模型，下例配套 DeepSeek-V3。注意 `model.multi_latent_attention: True` 是 Muon 能构造的前提：

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 2
  max_norm: 1.0
  seed: 42
  deterministic: False

optimizer:
  type: Muon
  weight_decay: 0.1
  momentum: 0.95
  matched_adamw_rms: 0.2
  nesterov: True
  eps: 1.e-7
  ns_steps: 5
  ns_coefficients: [3.4445, -4.7750, 2.0315]
  adamw_betas:
    - 0.95
    - 0.95
  adamw_eps: 1.e-8
  qk_clip_enabled: True
  qk_clip_threshold: 100
  comm_strategy: allgather          # 多卡可改用 allgather_deredundency 去冗余

lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 2.e-2
  warmup_ratio: 0.0
```

---

## 五、相关文档

- [配置文件说明](./configuration.md)：各配置段的总览与解析规则。
- [分布式并行训练](./parallel_training.md)：数据并行度、`global_batch_size` 与梯度累积步数的推导。
- [其它训练特性](./other_training_features.md)：梯度累积、检查点等训练特性。
- [训练启动指南](../guide/training.md)：如何用 `run_mindformer.py` 启动训练。
- [概述](../introduction/overview.md)：MindSpore Transformers 动态图整体介绍。
- [快速开始](../quick_start/quick_start.md)：端到端跑通一次训练。