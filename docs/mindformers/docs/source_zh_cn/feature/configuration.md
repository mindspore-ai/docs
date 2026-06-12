# 配置文件说明

动态图（PyNative）训练使用一个 YAML 文件集中管理所有可配置项，由 [mindformers/pynative/config/config.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/config/config.py) 的 **dataclass 配置体系**（`TrainConfig` 及其子配置类）在对 YAML 文件加载时进行解析与校验：

- YAML 顶层每一段对应 `TrainConfig` 的一个字段，映射到一个子配置类；
- 加载时这些子配置类会对配置项进行类型转换与合法性检查，校验失败直接抛错（如 `global_batch_size` 必须为正）；
- 标注 `allow_extra = True` 的配置类允许写入类中未声明的额外字段，框架原样透传给底层模块（如模型结构超参、调度器超参）。

---

## 顶层段与对应特性页速查

一份完整配置由以下 14 个顶层段组成（对应 `TrainConfig` 的字段）。下表给出每段的配置类、是否允许额外字段（`allow_extra`），以及深入说明所在的特性页：

| 顶层段              | 配置类                    | `allow_extra`              | 详见页面                                                     |
|------------------|------------------------|----------------------------|----------------------------------------------------------|
| `checkpoint`     | `CheckpointConfig`     | 否                          | [权重保存与加载](./safetensors.md)、[断点续训](./resume_training.md) |
| `training`       | `TrainingConfig`       | 否                          | 本章节                                                      |
| `parallelism`    | `ParallelismConfig`    | 否                          | [分布式并行训练](./parallel_training.md)                        |
| `optimizer`      | `OptimizerConfig`      | **是**                      | [训练超参数与优化器](./training_hyperparameters.md)               |
| `lr_scheduler`   | `LrSchedulerConfig`    | **是**                      | [训练超参数与优化器](./training_hyperparameters.md)               |
| `train_dataset`  | `TrainDatasetConfig`   | 否（其中 `dataloader` 为 **是**） | [数据集](./dataset.md)                                      |
| `model`          | `ModelConfig`          | **是**                      | 本章节 + 模型配置README                                         |
| `monitor`        | `MonitorConfig`        | 否                          | [训练指标监控](./monitor.md)                                   |
| `profiler`       | `ProfilerConfig`       | 否                          | [训练指标监控](./monitor.md)（Profiling 小节）                     |
| `recompute`      | `RecomputeConfig`      | 否                          | [训练内存优化](./memory_optimization.md)                       |
| `recompute_comm` | `RecomputeCommConfig`  | 否                          | [训练内存优化](./memory_optimization.md)                       |
| `swap`           | `SwapConfig`           | 否                          | [训练内存优化](./memory_optimization.md)                       |
| `callbacks`      | `List[CallbackConfig]` | 列表项为 **是**                 | 本章节                                                      |

- `allow_extra = False`：写入未声明字段会直接报 `Unknown configuration keys`。
- `allow_extra = True`：未声明字段被原样透传给底层（如优化器实现、模型类、调度器、Callback 注册器）。这意味着字段表里列出的是 **固定字段**，其余字段是 **按类型扩展的透传字段**，需查对应实现确认其名称与含义。

每个子配置类都有默认值，因此 **绝大多数段可省略**，省略时整段使用默认配置（如 `recompute`、`swap`、`profiler` 不写即关闭）。必须显式提供的只有：`model`（模型类型与结构）、`train_dataset`（数据来源）、以及训练规模相关的 `training`。

---

## 完整 YAML 骨架

下面是一份从 `checkpoint` 到 `callbacks` 的端到端、可直接保存运行的最小完整 `train.yaml`。注释标注了哪些段可省略走默认。模型段以 DeepSeek-V3 为例，实际请换成目标模型的结构超参。

```yaml
# ===== 权重保存与加载（可省略，省略则按默认保存到 save_path）=====
checkpoint:
  # 保存配置，默认
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  # 加载配置
  load_path: ""                 # 默认为空，表示不加载权重
  no_load_optim: False          # 断点续训务必为 False

# ===== 训练规模（建议显式提供）=====
training:
  steps: 1000
  local_batch_size: 1
  global_batch_size: 8
  max_norm: 1.0
  seed: 42

# ===== 多维并行（可省略，省略则纯 FSDP 单策略）=====
parallelism:
  data_parallel_shard: -1                      # -1 表示自动按可用卡数切分
  tensor_parallel: 1
  pipeline_parallel: 1
  context_parallel: 1
  expert_parallel: 1
  # sequence_parallel: False # 随 TP 自动开启，当前不支持关闭（无需配置此项，详见并行训练文档第六节）

# ===== 优化器（type 固定，其余按优化器类型透传）=====
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01

# ===== 学习率调度（type/learning_rate 固定，warmup 等按调度器类型透传）=====
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0                              # 透传字段，含义见具体学习率调度器的实现

# ===== 数据集（必须提供数据来源）=====
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: False
    config:                    # 完整字段见「数据集」文档
      seq_length: 4096
      split: "1, 0, 0"
      eod: 1
      pad: -1
      create_attention_mask: False
      create_compressed_eod_mask: False
      data_path:
        - '1'
        - "/path/megatron_data_text_document"
  drop_remainder: True
  num_parallel_workers: 8

# ===== 模型（必须提供）：model_type/architectures 固定，其余结构超参透传 =====
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  vocab_size: 129280
  seq_length: 4096
  hidden_size: 1792
  num_hidden_layers: 12
  num_attention_heads: 8
  use_flash_attention: True
  compute_dtype: "bfloat16"

# ===== 以下段均可整段省略，省略即关闭对应能力 =====
monitor:
  moe_monitor:
    save_tokens_per_expert_interval: 100
    target_layers: [0, 1, 2]

profiler:
  enable_profiling: False

recompute:
  mode: "None"

swap:
  enable: False

callbacks: []
```

将上面内容保存为 `train.yaml` 后，按 [加载与启动](#加载与启动) 一节的命令即可拉起训练。

---

## checkpoint —— 权重保存与加载

**是什么**：控制权重与优化器状态的保存与加载，包括保存路径、保存频率、保留数量、是否异步、断点续训时的加载行为。

**何时用**：训练都需要它来落盘；断点续训、增量微调、跨卡均衡加载等场景需要重点调整 `load_path` 与 `no_load_optim`。

**怎么配**：

| 字段                         | 默认值            | 说明                               |
|----------------------------|----------------|----------------------------------|
| `enable_save`              | `True`         | 是否保存权重                           |
| `save_path`                | `""`           | 权重保存目录                           |
| `save_max`                 | `5`            | 最多保留的 checkpoint 数，超出按时间淘汰旧的     |
| `save_interleaved_steps`   | `1000`         | 每多少步保存一次                         |
| `no_save_optim`            | `False`        | 为 `True` 时只存模型权重、不存优化器状态（无法精确续训） |
| `async_save`               | `False`        | 异步保存，减少保存阻塞                      |
| `prefix`                   | `"checkpoint"` | 保存文件名前缀                          |
| `remove_redundancy`        | `False`        | 保存时去除分布式冗余数据，减小文件体积              |
| `load_path`                | `""`           | 加载目录；为空则不加载，从头训练                 |
| `load_balanced`            | `False`        | 跨卡均衡加载                           |
| `no_load_optim`            | `False`        | 为 `True` 时不加载优化器状态               |
| `load_worker_number`       | `1`            | 加载线程数                            |
| `save_global_layout_cache` | `True`         | 保存全局 layout 缓存，加速后续加载/恢复         |

>断点续训需要恢复优化器状态，必须保证 `no_load_optim: False`（默认）。若 `no_load_optim: True`，仅恢复模型权重，无法精确接续训练。

YAML 配置示例如下：

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_max: 3
  save_interleaved_steps: 500
  async_save: True
  prefix: "qwen3"
  load_path: "./output/ckpt"     # 断点续训：指向上次保存目录
  no_load_optim: False
```

字段细节见 [权重保存与加载](./safetensors.md)；续训流程见 [断点续训](./resume_training.md)。

---

## training —— 训练基础参数

**是什么**：训练步数、批大小、梯度裁剪、随机种子等运行规模参数。

**何时用**：每次训练都需要；调整吞吐/收敛行为时主要改这里。

**怎么配**：

| 字段                  | 默认值     | 说明                                                                                                                                |
|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| `steps`             | `1000`  | 总训练步数                                                                                                                             |
| `local_batch_size`  | `1`     | 每卡 batch size，必须为正                                                                                                                |
| `global_batch_size` | `1`     | 全局 batch size，必须为正；梯度累积步数由 `global_batch_size / (data_parallel × local_batch_size)` 推导（`data_parallel`会根据使用的卡数和其他并行配置自动推导，无需手动配置） |
| `max_norm`          | `1.0`   | 梯度裁剪的目标范数：全局梯度范数超过该值时按比例缩放梯度                                                                                                      |
| `seed`              | `42`    | 随机种子                                                                                                                              |
| `deterministic`     | `False` | 确定性训练，开启此开关可以确保多次重复相同任务的结果完全一致，但会造成性能下降                                                                                           |

`global_batch_size` 与 `local_batch_size` 都必须为正，否则加载时 `TrainingConfig.__post_init__` 直接抛错 `must be positive`。此外训练初始化时要求 `data_parallel × local_batch_size <= global_batch_size`（不满足直接抛错），并据此用整数除法推导梯度累积步数（`global_batch_size // (data_parallel × local_batch_size)`）；非整除不会报错，而是向下取整（有效全局批次相应变小）。

> 梯度裁剪在每个优化器步恒定执行，先求全局梯度范数，再用 `clip_coef = max_norm / (global_norm + eps)`，当 `clip_coef < 1` 时按比例缩放梯度。
`max_norm` 越大越不易触发缩放，但并不存在用非正值关闭裁剪的语义，请保持 `max_norm > 0`。

YAML 配置示例如下：

```yaml
training:
  steps: 2000
  local_batch_size: 2
  global_batch_size: 16
  max_norm: 1.0
  seed: 1234
  deterministic: False
```

---

## parallelism —— 多维并行

**是什么**：声明 FSDP/HSDP、张量并行（TP）、上下文并行（CP）、流水线并行（PP）、专家并行（EP）、序列并行（SP）等并行策略及其细化选项。

**何时用**：单卡/小规模可整段省略走默认（纯 FSDP）；多卡大模型按显存与吞吐需求组合各并行维度。

**怎么配**（`ParallelismConfig` 全部字段）：

| 字段                                    | 默认值                    | 说明                                                    |
|---------------------------------------|------------------------|-------------------------------------------------------|
| `data_parallel_shard`                 | `-1`                   | FSDP 切分数，`-1` 自动按可用卡数                                 |
| `data_parallel_shard_strategy`        | `"optim_grads_params"` | FSDP 切分策略                                             |
| `reshard_after_forward_policy`        | `"default"`            | FSDP 前向后是否重新分片：`always`/`never`/`default`             |
| `cpu_offload`                         | `False`                | FSDP 参数/状态 CPU offload                                |
| `disable_gradient_division`           | `True`                 | 关闭 FSDP 自动梯度均分（用 sum 而非 mean）                         |
| `tensor_parallel`                     | `1`                    | 张量并行度（TP）                                             |
| `context_parallel`                    | `1`                    | 上下文并行度（CP）                                            |
| `context_parallel_method`             | `"colossal"`           | CP 实现方法                                               |
| `context_parallel_async`              | `False`                | 启用 Hyper-Parallel 异步 CP hook                          |
| `ulysses_degree_in_cp`                | `None`                 | CP 内 Ulysses 度，混合 CP 时必填                              |
| `context_parallel_mask_type`          | `"causal"`             | CP 输入准备路径支持的注意力 mask 类型                               |
| `pipeline_parallel`                   | `1`                    | 流水线并行度（PP 阶段数）                                        |
| `pipeline_parallel_layers_per_stage`  | `None`                 | 每个 PP 阶段的层分配（支持不均匀/交错放置）                              |
| `pipeline_parallel_schedule`          | `"1f1b"`               | PP 调度策略                                               |
| `pipeline_parallel_interleave_num`    | `1`                    | 交错的 model chunk 数                                     |
| `pipeline_parallel_overlap_p2p`       | `False`                | PP 阶段通信 overlap                                       |
| `pipeline_parallel_overlap_b_f`       | `False`                | PP 阶段计算 overlap                                       |
| `pipeline_parallel_enable_dxdw_split` | `False`                | PP 中 dxdw 通信拆分                                        |
| `sequence_parallel`                   | `False`                | 序列并行（SP）。**当前 SPMD 路径不使用此字段**：开启 TP 即自动启用 SP，暂不支持单独关闭 |
| `expert_parallel`                     | `1`                    | 专家并行度（EP，MoE）                                         |
| `expert_tensor_parallel`              | `1`                    | 专家张量并行度                                               |
| `npu_nums_per_device`                 | `8`                    | 每设备的 NPU rank 数                                       |
| `moe_token_dispatcher_type`           | `"alltoall"`           | MoE token 分发方式：`alltoall` / `alltoall_deredundancy`   |

当 `moe_token_dispatcher_type: alltoall_deredundancy` 时，必须满足 `expert_parallel >= npu_nums_per_device`，否则 `ParallelismConfig.__post_init__` 抛错。`reshard_after_forward_policy` 仅接受 `always`/`never`/`default`，其它取值在 FSDP 初始化时报错。YAML 配置示例如下：

```yaml
# 示例：FSDP + TP + PP + MoE EP 组合
parallelism:
  data_parallel_shard: -1
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 2
  pipeline_parallel: 2
  pipeline_parallel_overlap_p2p: True       # PP overlap 系列
  pipeline_parallel_overlap_b_f: True
  context_parallel: 1
  expert_parallel: 8                        # MoE：EP >= npu_nums_per_device
  expert_tensor_parallel: 1
  npu_nums_per_device: 8
  moe_token_dispatcher_type: "alltoall_deredundancy"
```

各维度含义、组合约束与显存/吞吐取舍见 [分布式并行训练](./parallel_training.md)。

---

## optimizer —— 优化器

**是什么**：优化器类型与超参。`OptimizerConfig` 标注 `allow_extra = True`，`type` 之外的具体实现超参按优化器类型透传。

**何时用**：每次训练都需要；切换 AdamW/Muon 或调整 weight decay 行为时改这里。

**怎么配**（固定字段）：

| 字段 | 默认值 | 说明                                                              |
|---|---|-----------------------------------------------------------------|
| `type` | `"AdamW"` | 优化器类型（如 `AdamW`、`Muon`）                                         |
| `betas` | `[0.9, 0.95]` | 一阶/二阶动量系数                                                       |
| `eps` | `1.0e-8` | 数值稳定项                                                           |
| `weight_decay` | `0.01` | 权重衰减系数                                                          |
| `weight_decay_include` | `None` | 参数名规则列表，与此项匹配的参数强制开启 weight decay                               |
| `weight_decay_exclude` | `None` | 参数名规则列表，与此项匹配的参数强制关闭 weight decay                               |
| `accumulate_allreduce_grads_in_fp32` | `True` | 对 bf16/fp16 参数注册反向 hook，将梯度转换至 fp32 后累积，使梯度累积与优化器更新在 fp32 精度下进行 |

```yaml
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.1
  weight_decay_exclude: ["bias", "norm"]    # norm/bias 不做衰减
  accumulate_allreduce_grads_in_fp32: True
```

更多优化器与超参说明见 [训练超参数与优化器](./training_hyperparameters.md)。

---

## lr_scheduler —— 学习率调度

**是什么**：学习率调度策略。`LrSchedulerConfig` 仅声明 **两个固定字段**：`type` 与 `learning_rate`；其余字段（如 `warmup_ratio`、`warmup_steps`、`min_lr` 等）**按调度器类型透传**，并非固定字段。

**何时用**：每次训练都需要；切换 warmup/衰减策略时需要修改这里。

**怎么配**（固定字段）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `type` | `None` | 调度器类型（如 `ConstantWarmUpLR`、`CosineWithWarmUpLR` 等） |
| `learning_rate` | `1e-5` | 基础学习率 |

下面示例中的 `warmup_ratio` 是非固定字段，会透传给所选调度器 `ConstantWarmUpLR` 。

```yaml
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0          # 透传字段，按调度器类型解析
```

调度器类型清单与各自专有字段见 [训练超参数与优化器](./training_hyperparameters.md)。

---

## train_dataset —— 数据集

**是什么**：数据加载器与数据集配置。`train_dataset` 顶层字段对所有加载器生效；`dataloader` 段标注 `allow_extra = True`，Megatron 加载器（`BlendedMegatronDatasetDataLoader`）特有字段写在其中。

**何时用**：每次训练都需要提供数据来源。

**怎么配**：

| 字段                                  | 默认值                       | 说明                 |
|-------------------------------------|---------------------------|--------------------|
| `dataloader`                        | `DataloaderConfig()`      | 加载器子配置             |
| `dataloader.type`                   |                           | 加载器类型              |
| `dataloader.shuffle`                | `False`                   | 是否打乱数据             |
| `dataloader.column_names`           | `["input_ids", "labels"]` | 数据列名               |
| `dataloader.python_multiprocessing` | `False`                   | 使用python多进程        |
| `drop_remainder`                    | `True`                    | 丢弃不足一个 batch 的尾部样本 |
| `num_parallel_workers`              | `8`                       | 并行数据加载 worker 数    |
| `prefetch_size`                     | `1`                       | 预取批次数              |
| `numa_enable`                       | `False`                   | 启用 NUMA 亲和加载       |

`dataloader` 内固定字段为`type`、`shuffle`、`column_names`、`python_multiprocessing`，其余非固定字段需要按具体加载器类型透传。

以下示例配置可用于加载Megatron格式数据集：

```yaml
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: False
  drop_remainder: True
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: False
```

Megatron 数据集的制作、字段与场景化配置见 [数据集](./dataset.md)。

---

## model —— 模型

**是什么**：模型类型与结构超参。`ModelConfig` 仅 `model_type` 与 `architectures` 为固定字段，其余全部结构超参（`hidden_size`、`num_hidden_layers`、MoE 配置等）通过透传交给模型类。

**何时用**：每次训练都必须提供。

**怎么配**（固定字段）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `model_type` | `None` | 模型类型标识（如 `deepseek_v3`、`qwen3`） |
| `architectures` | `None` | 模型架构类名（如 `DeepseekV3ForCausalLM`） |

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  # 以下均为透传的结构超参，须与所选模型匹配
  vocab_size: 129280
  seq_length: 4096
  hidden_size: 1792
  intermediate_size: 3072
  num_hidden_layers: 12
  num_attention_heads: 8
  use_flash_attention: True
  compute_dtype: "bfloat16"
  # MoE 结构超参（如适用）
  n_routed_experts: 8
  num_experts_per_tok: 4
```

`model` 段除两个固定字段外都是透传项，字段名/默认值由所选模型类决定，写错的字段不会报 `Unknown keys`，但会在模型构造时被忽略或导致报错，请以目标模型的结构定义为准。

---

## monitor —— 训练监控

**是什么**：训练稳定性健康检查、训练状态监控、TensorBoard 日志、MoE 监控。各子项均有默认，整段可省略。

**何时用**：需要可视化/落盘训练指标，或排查训练异常（梯度爆炸、loss 异常、MoE 负载不均）时启用。

**怎么配**（子段与关键字段）：

| 子段 | 关键字段（默认值） | 说明 |
|---|---|---|
| `train_state` | `local_norm`(`""`)、`local_loss`(`False`)、`device_norm`(`False`)、`device_loss`(`False`) | 本地/设备级 norm 与 loss 监控 |
| `moe_monitor` | `save_tokens_per_expert_interval`(`None`)、`target_layers`(`None`) | MoE 每专家 token 数记录（`None` 关闭） |

yaml示例如下：

```yaml
monitor:
  train_state:
    local_loss: True
  moe_monitor:
    save_tokens_per_expert_interval: 100
    target_layers: [0, 1, 2]
```

字段细节见 [训练指标监控](./monitor.md)。

---

## profiler —— 性能分析

**是什么**：采集性能 profiling 数据（算子耗时、内存、调用栈等）。`enable_profiling` 默认关闭，整段可省略。

**何时用**：定位性能瓶颈、分析算子分布与显存占用时启用，并配合指定步区间避免数据量过大。

**怎么配**：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enable_profiling` | `False` | 是否开启 profiling |
| `start_step` | `1` | 起始步（必须为正） |
| `end_step` | `1` | 结束步（必须为正，且须 `>= start_step`） |
| `output_path` | `None` | 结果保存目录，`None` 时自动生成 |
| `profiler_rank` | `None` | 指定采集的 rank 列表，`None` 表示全部 rank |
| `profiler_level` | `0` | 采集级别（`0`/`1`/`2`，越大越详细） |
| `mstx` | `False` | 是否采集轻量级打点数据 |
| `profile_memory` | `False` | 是否采集 Tensor 内存数据 |
| `with_stack` | `True` | 是否采集 Python 侧调用栈 |

`start_step` 与 `end_step` 都必须为正，且必须满足 `start_step <= end_step`，否则 `ProfilerConfig.__post_init__` 直接抛错。

YAML 配置示例如下：

```yaml
profiler:
  enable_profiling: True
  start_step: 3
  end_step: 5
  profiler_level: 1
  profile_memory: True
  with_stack: True
  profiler_rank: [0]
  output_path: "./output/profile"
```

分析方法与生成文件解读见 [训练指标监控](./monitor.md) 的 Profiling 小节。

---

## recompute —— 重计算

**是什么**：activation checkpoint（重计算），用前向重算换显存。PyNative 下通过对指定层/模块包裹 checkpoint wrapper 实现。

**何时用**：显存吃紧时启用；按层全量重算（`full`）或只对部分模块重算（`select`）。

**怎么配**：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mode` | `"None"` | 重计算模式：`"None"` / `"full"` / `"select"` |
| `full_recompute_layer` | `None` | `mode: full` 时必填，**层 spec 字符串列表**（如 `['0-3']`、`['0','5']`），按层序号升序、不重叠 |
| `select_module` | `None` | `mode: select` 时必填，**字典**：模块路径 → 层 spec 字符串列表 |

`full_recompute_layer` 与 `select_module` 的值都是 **字符串 spec 列表**，单个序号写 `'5'`，区间写 `'0-3'`（闭区间，含端点，即第 0、1、2、3 层共 4 层）。请加引号写成 `['0-3']`，避免裸 `[0-3]` 被 YAML 误解析；spec 必须严格升序且不重叠，最大层号需小于模型层数（合法范围 `[0, num_layers-1]`）。

`mode: full` 示例（按层全量重算）：

```yaml
recompute:
  mode: "full"
  full_recompute_layer: ['0-3']     # 前 4 层全量重算
```

`mode: select` 示例（按模块选择性重算）：

```yaml
recompute:
  mode: "select"
  select_module:
    "attention": ['0-1']            # 第 0、1 层的 attention 模块重算
    "mlp": ['2']                    # 第 2 层的 mlp 模块重算
```

策略选择与显存收益见 [训练内存优化](./memory_optimization.md)。

---

## recompute_comm —— 通信重计算

**是什么**：对通信算子做选择性重计算，进一步降低显存峰值。与 `recompute.mode` 相互独立，由自身 `enable` 单独控制。

**何时用**：在常规重计算之外仍需压显存，且通信算子保存的中间值占用可观时启用。

**怎么配**：

| 字段 | 默认值 | 说明                                                          |
|---|---|-------------------------------------------------------------|
| `enable` | `False` | 是否开启通信重计算                                                   |
| `select_module` | `None` | `enable: True` 时必填，否则直接抛错，格式为**字典**：通信算子模块路径 → 层 spec 字符串列表 |

示例如下：

```yaml
recompute_comm:
  enable: True
  select_module:
    "attention.qkv": ['0-3']
```

详见 [训练内存优化](./memory_optimization.md)。

---

## swap —— 激活值 SWAP（offload）

**是什么**：将 transformer block 或指定算子的激活值 offload 到 CPU，反向时再取回，以带宽换显存。

**何时用**：显存严重不足、重计算仍不够时启用。代价是反向时从 CPU 取回数据的延迟。

**怎么配**：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enable` | `False` | 是否开启 transformer block / 算子 offload |
| `default_prefetch` | `1` | 反向 FlashAttention 之前预取激活值的算子提前量，用于掩盖 CPU→NPU 取回延迟 |
| `layer_swap` | `None` | 整层 offload 配置列表，每项形如 `{layers: ['<spec>']}` |
| `op_swap` | `None` | 算子级 offload 配置列表，每项形如 `{op_name: <模块名>, layers: ['<spec>']}` |

`default_prefetch` 在动态图下用于在执行反向 FlashAttention 前提前发起激活值取回，掩盖 CPU→NPU 数据搬运延迟；其值表示提前的算子数，取值范围 `[1, num_layers-1]`（越界会校验报错）。`layer_swap`/`op_swap` 中的 `layers` 同样是带引号的层 spec 字符串列表。

整层 offload 示例：

```yaml
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: ['0-1']          # offload 第 0、1 层
```

算子级 offload 示例：

```yaml
swap:
  enable: True
  op_swap:
    - op_name: "attention"
      layers: ['0']            # 第 0 层的 attention 算子 offload
    - op_name: "mlp"
      layers: ['1']            # 第 1 层的 mlp 算子 offload
```

详见 [训练内存优化](./memory_optimization.md)。

---

## callbacks —— 训练回调

**是什么**：训练回调列表（`List[CallbackConfig]`）。每个列表项是一个回调配置，`type` 为必填字段，其余字段按回调类型透传。框架内置回调始终生效，这里配置的是额外自定义回调。

**何时用**：需要在训练过程中插入自定义逻辑（打印、保存、监控等）时使用；不需要可写 `callbacks: []` 或整段省略。

**怎么配**：

| 字段 | 默认值 | 说明                              |
|---|---|---------------------------------|
| `type` | `None` | 回调类型，对应已注册到 `CALLBACK` 模块的类名，必填 |
| 其余字段 | — | 按回调类型透传给其构造函数                   |

```yaml
callbacks:
  - type: MyCustomCallback
    log_interval: 10          # 透传给该回调的构造参数
```

框架会对每个列表项调用注册器实例化对应回调，`type` 必须是已注册的回调类名。

---

## 加载与启动

动态图训练任务统一通过[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)拉起，执行单卡任务可参考如下命令：

```bash
python run_mindformer.py --config train.yaml --mode 1 --use_parallel False
```

多卡任务可通过scripts目录下的[msrun_launcher.sh](https://atomgit.com/mindspore/mindformers/blob/master/scripts/msrun_launcher.sh)脚本执行，参考如下命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config train.yaml --mode 1" 8 # 8表示8卡任务
```

最小可跑示例与启动细节见 [快速开始](../quick_start/quick_start.md)。

---

## 相关文档

- 概念与上手：[模型概述](../introduction/overview.md)、[快速开始](../quick_start/quick_start.md)
- 权重与续训：[权重保存与加载](./safetensors.md)、[断点续训](./resume_training.md)
- 数据与并行：[数据集](./dataset.md)、[分布式并行训练](./parallel_training.md)
- 超参与显存：[训练超参数与优化器](./training_hyperparameters.md)、[训练内存优化](./memory_optimization.md)
- 监控与分析：[训练指标监控](./monitor.md)（含 Profiling）
