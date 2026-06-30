# 分布式并行训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/parallel_training.md)

当模型规模超出单卡容量时，动态图（PyNative）训练采用 **多维混合并行**，把模型与数据按不同维度切分到多张卡上协同训练。可按需组合多种切分方法：按数据切分（数据并行 / FSDP）、按算子内权重切分（张量并行 TP）、按序列切分（上下文并行 CP）、按模型层切分（流水线并行 PP）、按 MoE 专家切分（专家并行 EP）。所有并行配置都写在配置文件的 `parallelism` 字段。

这些并行能力由 [HyperParallel](https://atomgit.com/mindspore/hyper-parallel/) 提供，是动态图训练的必需依赖（要求 MindSpore >= 2.10）。安装方式见[安装指南 · 安装 HyperParallel](../installation.md#安装-hyperparallel动态图训练必需)。

本页的使用顺序如下：

1. **选维度**：在「并行能力」中，根据模型规模与显存情况选择需要开启的并行维度；
2. **配参数**：参考「并行配置方法」，设置并行度与批大小两类参数；
3. **校验启动**：参考各维度的场景示例完成 YAML 配置，确认满足「并行配置约束」后启动训练。

## 并行能力

下表列出动态图支持的并行维度，可先根据模型规模与显存情况选择需要开启的维度：

| 维度          | 配置字段                            | 切分对象                               | 典型场景                          |
|-------------|---------------------------------|------------------------------------|-------------------------------|
| 数据并行 / FSDP | `data_parallel_shard`           | 样本维 + 参数/梯度/优化器状态分片                | **几乎所有训练的基础**；显存紧张时增大分片度      |
| HSDP（副本+分片） | `data_parallel_shard`（小于数据并行维度） | 节点内分片、跨节点复制                        | 多节点训练，需用节点内带宽做分片、节点间做复制       |
| 张量并行 TP     | `tensor_parallel`               | 算子内权重（注意力/FFN 线性层）                 | 单层权重无法在单卡容纳的大模型               |
| 上下文并行 CP    | `context_parallel`              | 序列维（含注意力计算）                        | **长序列**（长上下文/长文档），激活显存超出单卡容量时 |
| 流水线并行 PP    | `pipeline_parallel`             | 模型按层切到多个 stage                     | 层数较多、单卡无法容纳整个模型；常与 TP/DP 组合   |
| 专家并行 EP     | `expert_parallel`               | MoE 专家分布到不同卡                       | MoE 模型（DeepSeek-V3 等）         |
| 序列并行 SP     | （随 TP 自动开启）                     | TP 未切分部分（LayerNorm/Dropout/残差）的序列维 | 开启 TP 即自动生效，无需也无法单独配置         |

## 并行配置方法

分布式并行配置分为**并行度**与**批大小**两类参数，分别写在配置文件的不同字段中。

**① 并行度（写在 `parallelism` 字段）**

按「并行能力」表选出要开启的维度，设置对应的并行度；未开启的维度保持默认 `1`。

| 配置项                   | 说明                                                                       |
|-----------------------|--------------------------------------------------------------------------|
| `data_parallel_shard` | 数据并行分片度。默认 `-1`，表示用剩余卡全部做分片（纯 FSDP）；设为小于数据并行维度的正数则形成 HSDP（节点内分片 + 跨节点复制） |
| `tensor_parallel`     | 张量并行度                                                                    |
| `context_parallel`    | 上下文并行度                                                                   |
| `pipeline_parallel`   | 流水线并行度                                                                   |
| `expert_parallel`     | 专家并行度（仅 MoE 模型需要）                                                        |

**② 批大小（写在 `training` 字段）**

| 配置项                 | 说明                       |
|---------------------|--------------------------|
| `global_batch_size` | 一个训练步（一次优化器更新）累计消费的样本总数  |
| `local_batch_size`  | 每张卡一次前向处理的样本数            |

配好后，框架会自动推导数据并行维度、梯度累积步数等数值，并校验总卡数与批大小是否匹配——这些推导与约束规则统一见文末「[并行配置约束](#并行配置约束)」。

## 全局配置示例与字段汇总

以下是一份覆盖主要 `parallelism` 字段的配置总览，便于对照字段含义；实际使用时，按下文各维度的场景示例裁剪即可。

```yaml
parallelism:
  # —— 数据并行 / FSDP / HSDP ——
  data_parallel_shard: -1                 # 用户唯一需要配的 DP 字段；-1=剩余卡全部做分片(纯 FSDP)
  reshard_after_forward_policy: "default" # 前向后是否重新分片参数：always / never / default
  cpu_offload: False                      # 将分片后的参数/梯度卸载到 CPU
  disable_gradient_division: True         # 关闭 FSDP 自动梯度均分(按 sum 聚合)

  # —— 张量并行 TP ——
  tensor_parallel: 1

  # —— 上下文并行 CP ——
  context_parallel: 1
  context_parallel_method: "colossal"     # colossal / ulysses / hybrid
  context_parallel_async: False
  ulysses_degree_in_cp: null              # ulysses / hybrid 时按规则取值
  context_parallel_mask_type: "causal"

  # —— 流水线并行 PP ——
  pipeline_parallel: 1
  pipeline_parallel_interleave_num: 1     # 每个 stage 的交织模型块数
  pipeline_parallel_layers_per_stage: null   # pp>1 时必填：auto（均分）或各 stage 层区间列表，见 4.4
  pipeline_parallel_overlap_p2p: False
  pipeline_parallel_overlap_b_f: False

  # —— 专家并行 EP（MoE）——
  expert_parallel: 1                      # 须整除 dp_shard*cp*tp（专家分片网格 efsdp 由此计算得出）
  moe_token_dispatcher_type: "alltoall"   # alltoall / alltoall_deredundancy
  npu_nums_per_device: 8

  # —— 序列并行 SP ——
  # SP 随 TP 自动开启，当前不支持单独关闭，无需在此配置（见第六节）
```

## 一、数据并行与 FSDP / HSDP

### 1.1 概述

动态图默认采用 **FSDP（全分片数据并行）** 方案：在分片组内对参数、梯度、优化器状态做切分，前向/反向时按需 all-gather 权重、reduce-scatter 梯度。框架进一步把数据并行拆成两层：

- **`dp_shard`（分片度）**：组内对参数/梯度/优化器状态做切分——节省显存，组内有 all-gather/reduce-scatter 通信。
- **`dp_replicate`（副本度）**：组间整份复制权重、只在反向同步梯度——经典数据并行，通信少但不节省显存。

两者都大于 1 时即 **HSDP（混合分片数据并行）**：节点内分片、跨节点复制，兼顾节省显存与降低跨节点通信。只要 `dp_shard > 1` 或 `cp > 1`，FSDP 就会启用。

### 1.2 适用场景与选型

- **基线**：单机多卡训练直接用纯 FSDP（`data_parallel_shard: -1`，副本度为 1），把全部数据并行卡用于分片，显存利用率最高。
- **显存不足**：优先增大分片度（即让更多卡加入同一分片组）。显存极度紧张时再叠加 `cpu_offload`、`reshard_after_forward_policy: always`。
- **多节点希望降低跨节点通信**：使用 HSDP——把分片度设为「节点内卡数」，让框架推导出「副本度 = 节点数」，分片通信通过节点内高带宽完成，副本只做梯度 all-reduce。配置上只需把 `data_parallel_shard` 设成一个小于数据并行维度的正数。

### 1.3 字段表

| 参数名称                           | 数据类型  | 是否可选 | 默认值         | 取值说明                                                                                                                                           |
|--------------------------------|-------|------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_parallel_shard`          | int   | 可选   | `-1`        | 分片度（dp_shard）。`-1` 表示用剩余卡数全部做分片（纯 FSDP，副本度为 1）；填正整数则框架据此推导副本度，可形成 HSDP。                                                                        |
| `reshard_after_forward_policy` | str   | 可选   | `"default"` | 前向后是否重新分片参数：`always` / `never` / `default`。`never` 用显存换通信（前向后保留聚合权重，反向少一次 all-gather）；`always` 反之。`default` 等价于「非 PP 时 `always`、PP 时 `never`」。 |
| `cpu_offload`                  | bool  | 可选   | `False`     | 将分片后的参数/梯度卸载到 CPU 内存，进一步节省显存，代价是 H2D/D2H 拷贝。                                                                                                   |
| `disable_gradient_division`    | bool  | 可选   | `True`      | 关闭 FSDP 自动梯度均分：以 **sum** 而非 mean 聚合梯度（对所有 HSDP 子模块设 reduce op = "sum"）。                                                                        |

**常见问题**：

- `disable_gradient_division=True` 是默认值，意味着 FSDP 梯度按 **sum** 聚合而非取平均。由此带来的数量级差异，框架会在反向传播的 loss 缩放（系数 `1/(world_size//pp)`）与梯度范数计算中自动修正，无需用户手动设置任何缩放系数。
- `reshard_after_forward_policy: default` 在 **PP 下解析为 `never`**：因为 PP 每个 microbatch 都会重复前向，若每次都 reshard 会带来反复 all-gather，故默认保留聚合权重。

### 1.4 场景配置

#### 场景 A：单机 8 卡纯 FSDP（推荐基线）

8 卡全部做分片，副本度为 1。满足 `dp_replicate(1) * dp_shard(8) * cp(1) * tp(1) * pp(1) == 8`。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 8
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
  expert_parallel: 1
```

#### 场景 B：双机 16 卡 HSDP（节点内分片 + 跨节点复制）

每节点 8 卡，希望节点内分片、节点间复制。把分片度设为 8，框架推导出 `data_parallel = 16`、`dp_replicate = 16 // 8 = 2`、`dp_shard = 8`。满足 `2 * 8 * 1 * 1 * 1 == 16`。

```yaml
parallelism:
  data_parallel_shard: 8           # 节点内 8 卡分片；副本度 2 由框架推导
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
```

## 二、张量并行（TP）

### 2.1 概述

`tensor_parallel` 指定 TP 规模，对注意力与 FFN 中的线性层权重在 **算子内** 做列/行切分（`apply_non_moe_tp` 用 `ColwiseParallel` / `RowwiseParallel`），组内通过 all-gather / all-reduce 同步激活。

### 2.2 适用场景与选型

- 单层权重无法在单卡容纳、或单卡注意力/FFN 激活过大时启用。
- TP 通信发生在 **每一层的前反向**，对带宽极敏感，**通常只在单节点内** 开（TP 组规模一般 ≤ 节点内卡数，如 2/4/8）。
- 与 FSDP 正交：`world_size = dp_total * tp * pp * cp`，开了 TP 后剩余卡数自动用于数据并行。

### 2.3 字段表

| 参数名称               | 数据类型 | 是否可选 | 默认值  | 取值说明              |
|--------------------|------|------|------|-------------------|
| `tensor_parallel`  | int  | 可选   | `1`  | 张量并行规模，`1` 表示关闭。  |

### 2.4 场景配置：单机 8 卡 DP + TP 微调

`tp=2`，剩余 `data_parallel = 8 // 2 = 4` 全部用于分片（`data_parallel_shard: -1` 推导出 `dp_shard = 4`）。满足 `1 * 4 * 1 * 2 * 1 == 8`。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 4
  tensor_parallel: 2
  context_parallel: 1
  pipeline_parallel: 1
  expert_parallel: 1
  reshard_after_forward_policy: "default"
  disable_gradient_division: True
```

## 三、上下文并行（CP）

### 3.1 概述

`context_parallel` 在 **序列维** 对输入进行切分，使每张卡只处理一段序列，显著降低长序列下的激活显存与注意力计算量；FlashAttention 在 CP 组内交换 KV 完成全序列注意力。动态图支持三种实现方法 `context_parallel_method`：

| 方法         | 原理                         | 约束（`_validate_cp_method`）                                                                                                                               |
|------------|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `colossal` | 默认；按序列分块、组内交换 KV           | 无额外约束；`ulysses_degree_in_cp` 内部按 1 处理                                                                                                                   |
| `ulysses`  | 在注意力 head 维做 all-to-all 切分 | 要求 `ulysses_degree_in_cp == context_parallel`（不设则默认取 `context_parallel`）；`context_parallel_async=True` 时还要求 `num_attention_heads % ulysses_degree == 0` |
| `hybrid`   | colossal + ulysses 混合      | 必须设 `ulysses_degree_in_cp`，且满足 `1 < ulysses_degree_in_cp < context_parallel` 且 `context_parallel % ulysses_degree_in_cp == 0`                           |

### 3.2 适用场景与选型

- 主要用于 **长序列/长上下文**（`seq_length` 很大、激活显存超出单卡容量）。CP 组规模越大，单卡序列越短、激活显存越省，但组内 KV 交换通信也越多。
- `colossal`：通用首选，序列切分直接，无 head 整除限制。
- `ulysses`：在 head 维切分，更适合 head 数较多的模型；启用 async 时要保证 head 数能被 ulysses 度整除。
- `hybrid`：把 CP 拆成「序列分块 × head 切分」两层，适合既需切分长序列、又需 head 维并行的超长序列场景。

**CP 与 FSDP 的耦合**：启用 CP 时，**FSDP 也会作用于 CP 组**：分片网格名为 `fsdp`，规模为 `dp_shard * cp`，即便 `dp_shard == 1`（见 [parallel_dims.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/parallel_dims.py) 的 `fsdp_enabled` / `fsdp` 属性）。即启用 CP 会自动让参数在 CP 组内分片，无需单独配置。

此外，CP 在前向开始前会把一个 batch 的输入（`input_ids`、`position_ids`、掩码等）沿序列维切分并分发到 CP 组各卡，使每张卡只拿到自己负责的那段序列——这一步称为 **CP 输入准备**。它目前**仅支持** `context_parallel_mask_type: causal`，传入其它值会直接抛 `NotImplementedError`，也不接受用户自定义的 `attention_mask`：CP 依赖模型侧的**压缩注意力掩码**，必须按下文 3.3 开启掩码压缩。

### 3.3 注意力掩码压缩（CP 必需）

**`context_parallel > 1` 时必须开启注意力掩码压缩**，否则 [context_parallel.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/context_parallel.py) 的 `apply_context_parallel_model_io` 会直接抛错：

```text
Context parallel (context_parallel > 1) requires a compressed attention mask.
Please enable use_attn_mask_compression for non-eod data, or
create_compressed_eod_mask for eod data.
```

**必须压缩的原因（显存开销）**：未压缩时，注意力掩码是一张 `seq_length × seq_length` 的稠密布尔矩阵。CP 主要服务于**长序列**场景，`seq_length` 很大时这张稠密 mask 本身就会占用大量显存（与 CP「节省激活显存」的目标相悖），且各 CP rank 还需各自持有并切分它。压缩掩码只保留生成因果/EOD 掩码所需的紧凑信息（如各子序列长度），由模型在注意力算子内即时重建，**避免物化稠密矩阵**。因此 CP 强制要求压缩，并建议关闭数据集侧的稠密掩码构建。

按**数据集类型**分两种配法。「EOD 数据集」指：为提高吞吐，预训练常把多篇较短的文档**拼接（packing）**进同一条定长序列，文档之间用一个 **EOD（End-Of-Document，文档结束）特殊 token** 分隔；同时配 `reset_position_ids` / `reset_attention_mask`，使注意力**不跨越文档边界**（每篇文档内部各自做因果注意力）。这类数据集的掩码不再是单一的整段下三角，而是「分段块对角」结构，需用 `create_compressed_eod_mask` 记录各子序列长度（`actual_seq_len`）来紧凑表达。非拼接、一条序列即一篇文档的常规数据则属「非 EOD 数据集」，采用通用因果掩码压缩即可。

- **非 EOD 数据集**：在 YAML 的 **`model`** 字段开启 `use_attn_mask_compression`（通用因果掩码压缩）：

  ```yaml
  model:
    # ...
    use_attn_mask_compression: true      # 非 eod 数据集：模型侧开启因果掩码压缩
  ```

- **EOD 数据集**：在 YAML 的**数据集 `dataloader.config`** 字段开启 `create_compressed_eod_mask`，并把 `create_attention_mask` 设为 `false`（避免再额外构建稠密掩码、节省显存）：

  ```yaml
  train_dataset:
    dataloader:
      config:
        # ...
        create_compressed_eod_mask: true   # eod 数据集：生成压缩 EOD 掩码（同步开启模型侧 eod 掩码压缩）
        create_attention_mask: false       # 关闭稠密 attention_mask 构建，节省显存
  ```

  `create_compressed_eod_mask: true` 会被框架同步到模型侧的 `use_eod_attn_mask_compression`，满足 CP 的压缩要求；`create_attention_mask: false` 则确保数据管线不再物化那张 `seq_length × seq_length` 稠密掩码。

### 3.4 字段表

| 参数名称                         | 数据类型 | 是否可选 | 默认值          | 取值说明                                                                                                        |
|------------------------------|------|------|--------------|-------------------------------------------------------------------------------------------------------------|
| `context_parallel`           | int  | 可选   | `1`          | 上下文并行规模。                                                                                                    |
| `context_parallel_method`    | str  | 可选   | `"colossal"` | CP 实现方法：`colossal` / `ulysses` / `hybrid`。                                                                  |
| `context_parallel_async`     | bool | 可选   | `False`      | 启用异步 CP 通信钩子（Hyper-Parallel）；与 `ulysses` 同用时要求 head 数可被 ulysses 度整除。                                        |
| `ulysses_degree_in_cp`       | int  | 可选   | `None`       | Ulysses 维度。`ulysses` 时须等于 `context_parallel`；`hybrid` 时须 `1 < 值 < context_parallel` 且整除 `context_parallel`。 |
| `context_parallel_mask_type` | str  | 可选   | `"causal"`   | CP 输入准备（按序列维切分输入，见 3.2）所用的掩码类型，**仅支持 `causal`**，其它值直接抛 `NotImplementedError`。                               |

### 3.5 场景配置

> 以下 `parallelism` 示例均需配合 3.3 的掩码压缩配置（在 `model` 或数据集 `dataloader.config` 字段）一起使用，此处从略。

#### 场景 A：单机 8 卡 colossal CP（长序列）

`cp=2`，剩余 `data_parallel = 8 // 2 = 4` 做分片。FSDP 分片组规模 = `dp_shard(4) * cp(2) = 8`。满足 `1 * 4 * 2 * 1 * 1 == 8`。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 4
  tensor_parallel: 1
  context_parallel: 2
  context_parallel_method: "colossal"
  context_parallel_mask_type: "causal"
  pipeline_parallel: 1
```

#### 场景 B：ulysses CP（head 维切分）

`cp=4`，`ulysses` 要求 `ulysses_degree_in_cp == context_parallel`。满足 `1 * 2 * 4 * 1 * 1 == 8`（`data_parallel = 8//4 = 2`）。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 2
  context_parallel: 4
  context_parallel_method: "ulysses"
  ulysses_degree_in_cp: 4          # 必须等于 context_parallel
  context_parallel_async: False    # 若设 True，需保证 num_attention_heads % 4 == 0
  tensor_parallel: 1
  pipeline_parallel: 1
```

#### 场景 C：hybrid CP（序列分块 × head 切分）

`cp=4`，hybrid 要求 `1 < ulysses_degree_in_cp < cp` 且整除 `cp`，故取 2。满足 `1 * 2 * 4 * 1 * 1 == 8`。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 2
  context_parallel: 4
  context_parallel_method: "hybrid"
  ulysses_degree_in_cp: 2          # 1 < 2 < 4 且 4 % 2 == 0
  context_parallel_mask_type: "causal"
  tensor_parallel: 1
  pipeline_parallel: 1
```

## 四、流水线并行（PP）

### 4.1 概述

`pipeline_parallel` 把模型 **按层切分到多个 stage**，配合 microbatch 流水执行。动态图内部固定使用交织式 1F1B 调度器（`ScheduleInterleaved1F1B`），通过 interleave 把每个 stage 切成多个虚拟模型块以减小流水线气泡。

### 4.2 适用场景与选型

- 模型层数较多、单卡无法容纳整个模型时启用；常与 TP（节点内）、DP（节点间）组合成 3D 并行。
- 增大 `pipeline_parallel_interleave_num` 可进一步减小气泡，但会增加 stage 间 P2P 通信次数；可叠加 `pipeline_parallel_overlap_p2p` 把通信与计算重叠。

**微批数与调度策略（自动推导 / 预留）**：

- **`pipeline_parallel_microbatch_size` 由框架自动推导，不是用户可配项**。[trainer.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/trainer.py) 会用梯度累积步数覆盖它：`num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)`（其中 `data_parallel = dp_replicate * dp_shard` 为数据并行维度），随后 `pipeline_parallel_microbatch_size = num_accumulation_steps`。所以需通过 `global_batch_size` / `local_batch_size` 间接控制微批数（见[配置文件说明](./configuration.md)），手填该字段无效。
- **`pipeline_parallel_schedule`（`ParallelismConfig` 中默认 `"1f1b"`）当前未参与分支**：动态图固定使用上述交织式 1F1B 调度，该字符串不被读取，属预留项。真正影响流水行为的是 `pipeline_parallel_interleave_num` 与 `pipeline_parallel_overlap_p2p` / `pipeline_parallel_overlap_b_f`。
- **`pipeline_parallel_enable_dxdw_split`（默认 `False`）当前为预留项**：`ParallelismConfig` 中定义了该字段（用于 PP 中 dx/dw 通信拆分），但动态图代码中暂未读取使用，改动它不会改变行为，属预留/占位项，无需填写。

### 4.3 字段表

| 参数名称                                   | 数据类型     | 是否可选 | 默认值              | 取值说明                                                                                                     |
|----------------------------------------|----------|------|------------------|----------------------------------------------------------------------------------------------------------|
| `pipeline_parallel`                    | int      | 可选   | `1`              | 流水线 stage 数。                                                                                             |
| `pipeline_parallel_interleave_num`     | int      | 可选   | `1`              | 每个 stage 的交织模型块数（虚拟 stage 数 = `pp * interleave_num`），增大可减小气泡。                                            |
| `pipeline_parallel_layers_per_stage`   | list/str | 可选   | `None`           | 各 stage 的层分配。`pp > 1` 时**必填**：层数可被 `pp * interleave_num` 整除时配 `auto`（均匀放置）；否则配列表显式指定各 stage 的层区间（见 4.4）。 |
| `pipeline_parallel_overlap_p2p`        | bool     | 可选   | `False`          | 启用 stage 间 P2P 通信与计算重叠。                                                                                  |
| `pipeline_parallel_overlap_b_f`        | bool     | 可选   | `False`          | 启用反向(b)与前向(f)计算重叠。                                                                                       |
| `pipeline_parallel_microbatch_size`    | int      | 可选   | `1`（**自动推导**）    | 由框架自动推导，运行时被 `num_accumulation_steps` 覆盖，无需手填（见上方提示）。                                                    |
| `pipeline_parallel_schedule`           | str      | 可选   | `"1f1b"`（**预留**） | 预留项，动态图固定用交织 1F1B，当前不参与分支。                                                                               |
| `pipeline_parallel_enable_dxdw_split`  | bool     | 可选   | `False`（**预留**）  | 预留项，PP 中 dx/dw 通信拆分开关，当前动态图未读取使用，不影响行为。                                                                  |

### 4.4 场景配置

`pp > 1` 时必须通过 `pipeline_parallel_layers_per_stage` 指定各 stage 的层放置，按层数能否均分选择以下两种方式之一。

#### 场景一：层数可均匀放置时配置 `auto`

当 `num_hidden_layers` 能被虚拟 stage 总数（`pp * interleave_num`）整除时，配置 `auto` 即可，框架按虚拟 stage 顺序把层连续均分（每个虚拟 stage 分得 `num_hidden_layers / (pp * interleave_num)` 层）。例如 8 层模型、`pp=2`、`interleave_num=2`：4 个虚拟 stage 各放 2 层。

```yaml
parallelism:
  data_parallel_shard: -1
  tensor_parallel: 4
  pipeline_parallel: 2
  pipeline_parallel_interleave_num: 2     # 虚拟 stage 总数 = 2 * 2 = 4
  pipeline_parallel_layers_per_stage: auto
  pipeline_parallel_overlap_p2p: True
```

若层数不能整除而仍配置 `auto`，启动时会报错并提示改用显式配置。

#### 场景二：层数不能均分时显式指定各 stage 的层区间

列表每一项对应一个**物理 stage**（第 `i` 项给 `pp_rank = i`）；项内用逗号分隔出 `interleave_num` 段层区间，第 `k` 段是该卡的第 `k` 个交织模型块，对应虚拟 stage 编号 `k * pp + i`。区间写法为 `"start-end"`（闭区间）或单层 `"n"`，所有区间按虚拟 stage 顺序必须恰好覆盖 `0 ~ num_hidden_layers-1` 且层号递增。

以 10 层模型（层号 0–9）、`pp=2`、`interleave_num=2` 为例，`10 % 4 != 0` 无法均分，可显式配置：

```yaml
parallelism:
  data_parallel_shard: -1
  tensor_parallel: 4
  pipeline_parallel: 2
  pipeline_parallel_interleave_num: 2
  pipeline_parallel_layers_per_stage:
    - "0-1, 4-8"    # 物理 stage 0：虚拟 stage 0 放层 0-1，虚拟 stage 2 放层 4-8
    - "2-3, 9"      # 物理 stage 1：虚拟 stage 1 放层 2-3，虚拟 stage 3 放层 9
  pipeline_parallel_overlap_p2p: True
```

层的执行顺序依次为虚拟 stage 0、1、2、3，即层 0-1、2-3、4-8、9。各 stage 层数可以不等（如本例为 2/2/5/1），可用于平衡首尾 stage 上 embedding / lm_head 的额外显存与计算。

## 五、专家并行（EP）

### 5.1 概述

针对 MoE 模型（DeepSeek-V3 等），`expert_parallel` 把专家分布到不同卡，token 经 all-to-all 路由到所在卡的专家计算（`apply_moe_ep_tp`）。EP **不是独立的卡维度**——它不出现在 `dp_replicate * dp_shard * cp * tp * pp == world_size` 的乘积里，而是 **复用在 `dp_shard * cp * tp` 这片卡上**：[parallel_dims.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/parallel_dims.py) 的 `build_mesh` 为专家单独构造一张稀疏网格 `["pp", "dp_replicate", "efsdp", "ep"]`，专家的 FSDP 分片网格 `efsdp` 由下式计算得到：

```text
efsdp = dp_shard * cp * tp // expert_parallel
```

| 网格                      | 维度                                                                     | 适用对象                  |
|-------------------------|------------------------------------------------------------------------|-----------------------|
| dense（稠密）               | `["pp", "dp_replicate", "fsdp", "tp"]`，`fsdp = dp_shard*cp`            | 非专家参数（注意力 / 共享 FFN 等） |
| sparse（稀疏，`ep > 1` 时构造） | `["pp", "dp_replicate", "efsdp", "ep"]`，`efsdp = dp_shard*cp*tp // ep` | MoE 专家参数              |

因此 **`expert_parallel` 必须整除 `dp_shard * cp * tp`**（否则 `efsdp` 不是正整数，框架构建网格时报错）。`ep` 越大，`efsdp` 越小、单卡专家越少、显存越省，但 all-to-all 通信越多。专家是否随 TP 切分不需单独配置，由上述 `efsdp` 公式从 `tensor_parallel` / `expert_parallel` 的相对大小自动得到。

### 5.2 适用场景与选型

- **仅 MoE 模型需要**。专家数较多、单卡无法容纳全部专家时启用 EP；`expert_parallel` 越大，单卡专家越少、显存占用越低，all-to-all 通信越多。
- **专家是否随 TP 切分由 `ep` 与 `tp` 的相对大小自动决定**：当 `ep <= tp` 时 `efsdp = dp_shard*cp*tp//ep` 仍含 `tp` 的因子，相当于专家权重在 EP 之外还保留部分 TP 切分；当 `ep` 增大到将 `tp` 因子用尽（`ep` 接近 `cp*tp` 乃至更大）时，专家主要由 EP 分布、TP 切分弱化。约束始终是 `ep` 整除 `dp_shard*cp*tp`。
- **`alltoall` vs `alltoall_deredundancy`**：`alltoall` 是默认全连接 all-to-all 分发；`alltoall_deredundancy`（`DeredundancyExpertParallel`）用「节点间外层 all-gather/reduce-scatter + 节点内 all-to-all」的去冗余方案降低跨节点通信，**仅在多机大 EP 时收益明显**，并要求 `expert_parallel >= npu_nums_per_device`（否则 `ParallelismConfig.__post_init__` 直接报错）。

### 5.3 字段表

| 参数名称                        | 数据类型 | 是否可选 | 默认值          | 取值说明                                                                         |
|-----------------------------|------|------|--------------|------------------------------------------------------------------------------|
| `expert_parallel`           | int  | 可选   | `1`          | 专家并行规模；须整除 `dp_shard * cp * tp`（专家分片网格 `efsdp = dp_shard*cp*tp//ep` 由此计算得出）。 |
| `moe_token_dispatcher_type` | str  | 可选   | `"alltoall"` | token 分发方式：`alltoall` / `alltoall_deredundancy`。                             |
| `npu_nums_per_device`       | int  | 可选   | `8`          | 每节点 NPU 数，`alltoall_deredundancy` 约束所用。                                      |

### 5.4 场景配置

#### 场景 A：单机 8 卡 MoE，EP ≤ TP（专家保留部分 TP 切分）

`tp=4`、`ep=2`（`ep <= tp`）。剩余 `data_parallel = 8//(4*1*1) = 2`，故 `dp_shard = 2`。满足总规模 `1 * 2 * 1 * 4 * 1 == 8`；`ep(2)` 整除 `dp_shard*cp*tp = 2*1*4 = 8`，专家分片网格 `efsdp = 8//2 = 4`。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 2
  tensor_parallel: 4
  expert_parallel: 2               # 须整除 dp_shard*cp*tp(=8)
  moe_token_dispatcher_type: "alltoall"
  context_parallel: 1
  pipeline_parallel: 1
```

#### 场景 B：多机 16 卡 MoE，EP > TP + 去冗余分发

`tp=2`、`ep=8`（`ep > tp`）；用去冗余分发，要求 `ep(8) >= npu_nums_per_device(8)`。剩余 `data_parallel = 16//(2*1*1) = 8`，故 `dp_shard = 8`。满足总规模 `1 * 8 * 1 * 2 * 1 == 16`；`ep(8)` 整除 `dp_shard*cp*tp = 8*1*2 = 16`，专家分片网格 `efsdp = 16//8 = 2`（专家主要由 EP 分布，TP 切分被 EP 占用一部分）。

```yaml
parallelism:
  data_parallel_shard: -1          # 此时框架自动推导为 8
  tensor_parallel: 2
  expert_parallel: 8               # 须整除 dp_shard*cp*tp(=16)；并 >= npu_nums_per_device
  moe_token_dispatcher_type: "alltoall_deredundancy"
  npu_nums_per_device: 8           # 要求 expert_parallel >= 此值
  context_parallel: 1
  pipeline_parallel: 1
```

## 六、序列并行（SP）

序列并行（SP）在 TP 的基础上，对 TP 未切分的部分（LayerNorm、Dropout、残差）按 **序列维** 进一步切分，降低这些算子的激活显存。

**SP 随 TP 自动开启，当前不支持单独关闭**：动态图 SPMD 路径中，**只要开启 TP（`tensor_parallel > 1`）就会自动施加 SP**：源码 [base_models/gpt/parallelize.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/base_models/gpt/parallelize.py) 调用 `apply_non_moe_tp` 时形参 `enable_sp` 恒为 `True`，**不读取 `sequence_parallel` 配置字段**，因此当前无法通过配置关闭 SP。

> 注意：`sequence_parallel` 字段仍会被 [transformer_block.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/transformers/transformer_block.py) 的 `TransformerBlock.__init__` 读取，但**仅用于检测 CP 与 SP 的冲突并输出告警**（`cp > 1` 且 `sequence_parallel=True` 时提示「SP 与 CP 冲突，SP 被忽略」），它不是 SP 的功能开关。

因此**无需为 SP 单独配置**：按本页第二节配好 TP（`tensor_parallel > 1`），SP 即自动生效。

## 并行配置约束

设备网格的总规模必须满足以下整除关系，否则启动时框架会直接报错：

- **总规模**：`dp_replicate * dp_shard * cp * tp * pp == world_size`（`world_size` 即 `msrun` 的总卡数）。其中 `dp_replicate`（副本度）与 `dp_shard`（分片度）由框架自动推导，用户只配 `data_parallel_shard`。
- **数据并行维度**：`data_parallel = world_size // (tp * pp * cp)`，必须 ≥ 1，且能被推导出的 `dp_shard` 整除。
- **批大小**：`data_parallel * local_batch_size <= global_batch_size`，否则报错。框架据此推导梯度累积步数 `num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)`；开大并行度会减小 `data_parallel`，在批大小不变时累积步数随之变大。
- **TP / SP**：`tensor_parallel > 1` 才有意义；SP 随 TP 自动开启（无独立开关，当前不支持关闭，见第六节）。
- **CP 方法**：`ulysses` 要求 `ulysses_degree_in_cp == context_parallel`；`hybrid` 要求 `1 < ulysses_degree_in_cp < context_parallel` 且整除；`ulysses + async` 还要求 `num_attention_heads % ulysses_degree == 0`；CP 启用时 FSDP 分片组规模为 `dp_shard * cp`。
- **CP 掩码压缩（必需）**：`context_parallel > 1` 时必须开启压缩掩码——非 EOD 数据在 `model` 字段设 `use_attn_mask_compression: true`，EOD 数据在数据集 `dataloader.config` 字段设 `create_compressed_eod_mask: true` 且 `create_attention_mask: false`，否则报错「requires a compressed attention mask」（见 3.3）。
- **EP**：`expert_parallel` 必须整除 `dp_shard * cp * tp`（专家分片网格 `efsdp = dp_shard*cp*tp//ep` 须为正整数）；EP 不计入 `dp_replicate*dp_shard*cp*tp*pp==world_size` 的乘积；`moe_token_dispatcher_type = alltoall_deredundancy` 时要求 `expert_parallel >= npu_nums_per_device`。

## 启动

配好 `parallelism` 与批大小、确认满足上述并行配置约束后，用 `msrun` 拉起多卡训练：

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --join=True \
      run_mindformer.py --config your_config.yaml --mode 1
```

`--mode 1` 表示 PYNATIVE（动态图）；`--worker_num` 即总卡数 `world_size`，必须与配置推导出的 `dp_replicate * dp_shard * cp * tp * pp` 相等，否则启动即报错。

## 相关文档

- 配置文件总览、`global_batch_size` / `local_batch_size` 与梯度累积：[配置文件说明](./configuration.md)
- 数据按并行维度自动分片、每卡批大小推导：[数据集](./dataset.md)
- 显存相关的并行/重计算/offload 优化：[训练内存优化](./memory_optimization.md)
- 预训练中的并行场景实践：[训练指南](../guide/training.md)
- 端到端上手：[快速开始](../quick_start/quick_start.md)
