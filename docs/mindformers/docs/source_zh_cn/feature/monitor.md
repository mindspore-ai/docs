# 训练指标监控与 Profiling

MindSpore Transformers 动态图（PyNative）在训练循环中提供两类可观测能力：

- **训练指标监控**（`monitor`）：按 micro-step 采集梯度 / 参数范数（norm）、损失（loss）等细粒度指标（另含 MoE 专家负载监控的预留配置）。这些指标**当前仅通过训练日志（`logger.info`）输出**。
- **性能数据采集**（`profiler`）：在指定 step 区间内采集算子、通信、内存、调用栈等性能数据，结果以 MindSpore Profiler 产物落盘，用于性能分析与优化。

两类能力均通过训练 YAML 配置开启，互不依赖，可单独使用。如果尚未一次训练，建议先阅读 [快速开始](../quick_start/quick_start.md)；配置文件整体结构见 [配置文件说明](./configuration.md)。

> **输出形式说明**
>
> 动态图监控指标（`monitor` 段）目前**只写日志**：每条记录由 `Monitor._flush_logger` 以 `{ key: value, ... }` 形式打印到训练日志。要把 norm/loss 曲线可视化，需要自行从日志解析。

---

## 一、训练指标监控

`monitor` 段下有两个子配置：`train_state`、`moe_monitor`。

监控由 `MonitorGroup` 在训练初始化时按配置实例化，只有「激活」的子监控才会被创建：`train_state` 在任一 norm/loss 开关打开时激活；`moe_monitor` 在 `save_tokens_per_expert_interval` 为正整数时激活。指标在每个训练 step 结束时由 `MonitorCallback`（`on_step_end`）统一 flush 到日志。

### 1.1 train_state（norm 与 loss 监控）

按 micro-step 采集本地（local）与设备（device）级的梯度范数与损失。四个开关相互独立，可任意组合。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `local_norm` | bool / str / list[str] | `""` | 监控本地梯度范数。`True` 监控全部可训练参数；字符串或字符串列表按参数名子串过滤（`"layers.0"` 匹配名字含该子串的参数）；`""`/`False` 关闭。 |
| `local_loss` | bool | `False` | 监控每个 micro-batch 的本地损失。 |
| `device_norm` | bool / str / list[str] | `False` | 监控设备级（累积）梯度范数，取值含义同 `local_norm`。 |
| `device_loss` | bool | `False` | 监控设备级损失（梯度累积/all-reduce 之后的损失）。 |

> **device_norm 会强制采集 local_norm**
>
> 源码中 `device_norm` 的累积范数来自 `local_norm` 采集时缓存的 `prev_local_norms`。因此只要开启 `device_norm`，框架会在 `local_norm` 采集阶段一并计算所需参数的范数，即使 `local_norm` 本身为关闭状态。两者按各自的参数名过滤器独立筛选要打印的参数。

#### 输出解读

- **`local_norm` 是「本步增量范数」**。框架内部对每个参数维护累积范数 `accumulated`，按 micro-step 做差分：
    - 第 0 个 micro-step：`actual_norm = accumulated`；
    - 之后：`actual_norm = accumulated - prev`（`prev` 为上一 micro-step 的累积值）。
    这样每条记录反映的是**当前 micro-batch 对梯度范数的增量贡献**，而非到当前为止的总范数。
- **`device_norm` 是累积值**，直接打印 `prev_local_norms` 中缓存的累积范数。
- 日志逐参数打印，形如 `{ "local_norm": decoder.layers.0.self_attention.q_layernorm.weight:   0.000381 }`；同一 micro-step 的多条 norm 记录前会先打印一条形如 `{ "step": 4, "micro_step": 2 }` 作为分组标记。
- `local_loss` 记录形如 `{ "micro_step": 2, "local_loss": 0.123 }`；`device_loss` 在优化器更新后记录。

#### 使用场景

**场景 A：只看全局 loss 曲线（最简）**

如果只关心整体收敛、不需要逐参数范数，**不必开启 train_state 的任何 norm**。step 级 loss 由框架的 `LossCallback` 直接打印到日志，监控段可留空或仅开 `device_loss`：

```yaml
monitor:
  train_state:
    device_loss: True       # 在每个 step 优化器更新后打印归约后的损失
```

**场景 B：排查梯度爆炸 / 某层范数异常**

当出现 loss 尖刺、NaN 或怀疑某些层梯度异常时，用 `local_norm` 按参数名过滤定位问题层。增量范数语义（上文「输出解读」）让你能看到**是哪个 micro-batch、哪个参数**贡献了异常增量：

```yaml
monitor:
  train_state:
    # 只监控这几类参数的本步增量梯度范数，避免逐参数刷屏
    local_norm: ["attention", "layernorm"]
    local_loss: True        # 同时看每个 micro-batch 的本地损失，定位异常 micro-batch
```

> 提示：要监控全部参数用 `local_norm: True`；只看单一参数用字符串，如 `local_norm: "embedding"`。

**场景 C：device_norm 与 local_norm 联动观察累积**

需要同时看「每步增量」与「累积总量」时，同时开启两者。注意 `device_norm` 复用 `local_norm` 的采集结果（见上文提示块）：

```yaml
monitor:
  train_state:
    local_norm: True        # 每个 micro-step 的增量范数
    device_norm: True       # 梯度累积后的累积范数（强制触发 local_norm 采集）
    device_loss: True
```

---

### 1.2 moe_monitor（MoE 专家负载）

设计用于监控 MoE 解码层各专家分配到的 token 数（tokens-per-expert），用于诊断**专家负载不均**——若少数专家长期吃满、其余专家空转，说明路由失衡，需要调整负载均衡损失或路由策略。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_tokens_per_expert_interval` | int / None | `None` | 打印 tokens-per-expert 记录的 micro-step 间隔；`None` 关闭监控，须为正整数（写 bool 会报错）。 |
| `target_layers` | int / list[int] / None | `None` | 监控的目标层。`int` 表示前 N 层 `range(N)`，**仅匹配 decoder 层**；`list[int]` 表示指定层 id；`None` 表示自动发现模型中所有含 `tokens_per_expert` 的 MoE 模块。 |

> **target_layers 的 int 与 list 语义差异**
>
> - `int`（如 `3`）→ 展开为 `range(3)`，即模型第 0/1/2 层。
> - `list[int]`（如 `[0, 2, 4]`）→ 精确匹配这些层 id。
> - 若指定的某层 id 实际不是 MoE 层（没有 `tokens_per_expert`），框架会打印告警 `decoder.layers.X is not a MoE layer`。

#### 输出格式

MoE 监控以 **Megatron 的 tokens-per-expert JSON 格式**打印到日志（每行一条 JSON）。字段含义如下：

| 字段 | 含义 |
|------|------|
| `iter` | 全局 micro-step 计数（`global_micro_step`） |
| `step` | 训练 step id |
| `micro_step` | step 内的 micro-batch 序号 |
| `block` | `"decoder"` 或 `"mtp"` |
| `layer` | 层 id |
| `mtp_idx` | 仅 `block == "mtp"` 时出现，标识 MTP 子层 |
| `tpe` | 该 micro-batch 内各专家分到的 token 数列表（按专家差分得到的增量） |

#### 配置示例：MoE 负载不均诊断

以下为该监控的配置写法：

```yaml
monitor:
  moe_monitor:
    save_tokens_per_expert_interval: 10   # 每 10 个 micro-step 记录一次（设计语义）
    target_layers: [0, 5, 10]             # 抽样靠前/中/后几层，观察是否分层失衡
```

> 也可用 `target_layers: 4` 监控前 4 层，或省略该字段让框架自动发现全部 MoE 层（数据量会更大）。`save_tokens_per_expert_interval` 须为正整数，写 `True`/`False` 会在配置校验阶段报 `TypeError`。

---

### 1.3 MaxLogits 健康监测（callbacks: MaxLogitsMonitor）

大规模长时训练中，注意力 logits 数值溢出、梯度尖刺等会破坏训练稳定性。`MaxLogitsMonitor` 是动态图中**真正被消费的数值健康监测回调**，源码见 `mindformers/pynative/callback/max_logits_monitor.py`。与 `monitor` 段下的子配置不同，它通过 **`callbacks` 段**注册。

#### 是什么

在每个训练步收集模型各层的**最大注意力 logit**（max attention logit），分层打印到日志，并汇总输出全部层的均值（mean）与最大值（max），随后在每步结束时重置层内累计值，便于逐步观察注意力数值是否异常增大。

输出格式（`_dump`，`max_logits_monitor.py:88-108`）：

- 逐层一行，日志 tag 为 `max_attention_logit/<param_name>`，值为该层各项的列表（保留 4 位有效数字）；
- 汇总两行：`max_attention_logit/mean`、`max_attention_logit/max`；
- 每行前缀为 `step:[当前步/总步数]`，与 `TrainingStateMonitor` 的打印格式一致。

#### 何时用

- 怀疑注意力 logits 溢出 / 数值爆炸导致 loss 抖动或 NaN 时，用它定位是哪一层、在哪一步开始异常增大；
- 使用 Muon 优化器并启用 QK clip 时（此时框架会自动开启追踪，见下文场景），用它验证 clip 是否生效。

#### 怎么配

通过 `callbacks` 段注册回调即可：

```yaml
callbacks:
  - type: MaxLogitsMonitor
    step_interval: 1   # 每多少步输出一次，必须为正整数
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `step_interval` | int | `1` | 输出间隔步数；必须为**正整数**，否则在构造时抛 `ValueError`（`max_logits_monitor.py:51-54`）。 |

行为细节（`on_step_end`，`max_logits_monitor.py:58-78`）：

- 当 `step_interval > 1` 且当前步不是输出步时，**仅重置**累计值、不打印；
- 输出步先采集各层 `get_max_attention_logit()`，无数据则只重置；
- 无论是否打印，每步末都会调用 `reset_max_attention_logit()` 清零，保证下一步从干净状态开始统计。

#### 场景：配合 Muon + QK clip 自动开启追踪

模型侧是否采集 max attention logit，由 `model.track_max_attention_logit` 开关控制。框架在 `configure_max_logits_tracking`（`max_logits_monitor.py:120-132`）中自动判定是否需要开启：

- 优化器为 **Muon** 且 `qk_clip_enabled` 为真；**或**
- `callbacks` 中已配置 `MaxLogitsMonitor`。

满足任一条件时，`track_max_attention_logit` 被置为 `True`。该装配在 Trainer 初始化时触发（`pynative/trainer/trainer.py:153`），并在 `ensure_max_logits_reset_callback` 中确保存在一个 `MaxLogitsMonitor` 负责重置（`trainer.py:536-538`）。

因此在 **Muon + QK clip** 场景下，即使没有显式写 `MaxLogitsMonitor`，框架也会自动补一个用于重置；若想看到逐层日志，仍建议显式声明并设置 `step_interval`：

```yaml
# 场景：Muon + QK clip，显式开启逐层 max logit 观测
optimizer:
  type: Muon
  qk_clip_enabled: True       # 启用 QK clip，框架据此自动开启 track_max_attention_logit

callbacks:
  - type: MaxLogitsMonitor
    step_interval: 50         # 每 50 步打印一次逐层 max logit 与 mean/max
```

> **自动开启的是「追踪」，不是「打印」**
>
> Muon + QK clip 仅自动开启**模型侧追踪**并补一个负责重置的回调；要在日志中看到逐层数值，仍需显式配置 `MaxLogitsMonitor`（或接受默认 `step_interval=1` 的逐步输出）。

---

## 二、Profiling（性能数据采集）

`profiler` 段控制性能数据采集。采集仅在 `enable_profiling: True` **且**当前 rank 命中 `profiler_rank` 时生效，框架用 MindSpore `schedule` 在 `[start_step, end_step]` 区间内采集（区间外训练正常进行，不采集）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_profiling` | bool | `False` | 是否开启性能数据采集。 |
| `start_step` | int | `1` | 开始采集的 step（对应 `schedule` 的 `skip_first`），须为正整数。 |
| `end_step` | int | `1` | 结束采集的 step，须 ≥ `start_step`（采集 step 数 = `end_step - start_step + 1`）。如果设置了非默认的 `start_step`，务必同时设置 `end_step`，否则默认 `1` 可能导致区间无效。 |
| `output_path` | str / None | `None` | 结果保存路径，按 rank 分目录保存为 `output_path/rank_x`；`None` 时回退到 `当前工作目录/profile/rank_x`。 |
| `profiler_rank` | list[int] / None | `None` | 指定开启采集的 rank id 列表；`None` 表示所有 rank 均采集。 |
| `profiler_level` | int | `0` | 采集级别 `0` / `1` / `2`，级别越高越详细（映射到 `ProfilerLevel.Level0/1/2`，非法值回退 `LevelNone`）。 |
| `mstx` | bool | `False` | 是否开启轻量 mstx 打点（透传给 `_ExperimentalConfig`），用于在时间线上插入轻量标记。 |
| `profile_memory` | bool | `False` | 是否采集 Tensor 内存数据。 |
| `profile_cpu` | bool | `True` | 是否采集 CPU profiling 活动。 |
| `with_stack` | bool | `True` | 是否采集 Python 侧调用栈数据。 |

> **区间选取要避开首步预热**
>
> 第一个 step 通常包含编译/初始化等一次性开销，采集到的数据不代表稳态。建议把 `start_step` 设在若干步之后（如 `start_step: 5`），并只采集少量步（如 2~3 步），既避开预热又控制产物体积与开销。
>
> **开销取舍**
>
> `profile_memory: True` 与 `with_stack: True` 会显著增加采集开销和产物体积，可能拖慢被采集的几个 step。仅在排查内存/调用栈问题时开启；做纯算子耗时分析时建议关闭 `profile_memory`。

### 2.1 多卡下的 rank 选取

多卡训练时无需对所有卡采集——数据量大且彼此冗余。用 `profiler_rank` 指定要采集的 rank：

- 单机调优一般只采 `[0]`；
- 排查通信不均衡时，可对同一并行组内不同 rank 各采一个对照（如 `[0, 1]`）。
- rank 与并行维度（dp/tp/pp/cp）的对应关系见 [分布式并行训练](./parallel_training.md)，据此选取代表性的 rank。

### 2.2 输出目录与查看方式

- **目录结构**：配置了 `output_path` 时为 `output_path/rank_x/`；未配置（`None`）时为 `<cwd>/profile/rank_x/`。

### 2.3 场景化配置（按 profiler_level 三档）

#### 场景 A：轻量算子级分析（Level 0）

只看算子耗时分布、定位热点 kernel，开销最小。关闭内存与调用栈：

```yaml
profiler:
  enable_profiling: True
  start_step: 5            # 跳过前 4 步预热
  end_step: 7              # 采集 3 步
  profiler_rank: [0]
  profiler_level: 0
  profile_memory: False
  profile_cpu: False
  with_stack: False
  output_path: "./output/profile"
```

#### 场景 B：含通信分析（Level 1）

需要分析分布式训练中的通信耗时与计算/通信重叠时，提升到 Level 1，并按需对多个 rank 对照采集：

```yaml
profiler:
  enable_profiling: True
  start_step: 5
  end_step: 7
  profiler_rank: [0, 1]    # 同组内对照，观察通信不均衡
  profiler_level: 1
  mstx: True               # 轻量打点，便于在时间线上对齐关键阶段
  profile_memory: False
  with_stack: False
  output_path: "./output/profile"
```

#### 场景 C：含内存与调用栈（Level 2）

排查显存峰值或需要把耗时归因到 Python 调用栈时，开 Level 2 并打开内存与调用栈采集（开销最大，建议只采单卡、少量步）：

```yaml
profiler:
  enable_profiling: True
  start_step: 5
  end_step: 6              # 仅采集 2 步，控制产物体积
  profiler_rank: [0]
  profiler_level: 2
  profile_memory: True     # 采集 Tensor 内存
  with_stack: True         # 采集 Python 调用栈
  output_path: "./output/profile"
```

---

## 相关文档

- 日志解析与日志结构：[日志](./logging.md)
- 配置文件总览：[配置文件说明](./configuration.md)
- 启动训练任务：[启动任务](./start_tasks.md)
- 并行维度与 rank 选取（影响 `profiler_rank` 与分布式采集）：[分布式并行训练](./parallel_training.md)
- 训练全流程（含训练状态监控）：[训练指南](../guide/training.md)
- 第一个训练任务：[快速开始](../quick_start/quick_start.md)
