# 训练显存优化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/memory_optimization.md)

大模型训练中，**激活值（activation）** 通常是显存占用的主要来源。MindSpore Transformers 动态图（PyNative）提供多种显存优化功能，可在配置文件中独立或组合启用。本文介绍显存优化的概念与必要性、各类方法的原理，以及 MindSpore Transformers 支持的场景与配置方法。

## 一、显存优化概念与必要性

大模型训练的显存占用主要来源于四部分：**模型权重**、**优化器状态**、**梯度**、**激活值**。前三者可通过并行切分（TP/PP/EP）分摊到多卡，而激活值与序列长度、batch、层数强相关，**在大模型训练中通常是显存瓶颈**。

因此，显存优化的核心目标是在不改变数值结果的前提下，降低激活值的显存占用。本质是 **以其他资源换取显存**，主要有两种思路：

- **以算力换显存**：丢弃前向激活值，反向时重新计算 —— 即 **重计算（Recompute）**。
- **以 PCIe 带宽/延迟换显存**：保留激活值但搬运到 CPU 内存，反向前再预取回 NPU —— 即 **激活值 Swap**。

两者效果相近，区别在于代价不同：重计算消耗额外算力，Swap 消耗 PCIe 带宽与延迟。可根据集群资源余量选择，也可组合使用（分配到不同层）。

## 二、显存优化方法

### 重计算（Recompute）

[重计算](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/recompute.html)（Activation Checkpointing）在前向传播时丢弃部分中间激活值，反向传播时再重新计算所需激活值，以算力换取显存。

按丢弃与重算的粒度，分为两类：

- **激活重计算**：对模型层或层内模块做重计算。
  - `full`（完全重计算）：对指定整层重计算，整层激活值全部丢弃。显存收益最大，算力开销也最大。
  - `select`（选择重计算）：仅对指定模块（如 MLP）重计算。粒度细、开销小，适合精确控制范围。
- **通信重计算**：并行切分（TP/EP）引入的通信算子（如 AllGather、ReduceScatter）激活值占用较大时，单独对这些通信算子重计算，无需整层重计算。

> 通信重计算与激活重计算功能独立，可同时开启或关闭。

### 激活值 Swap

激活值 Swap 通过把激活值卸载到 CPU 内存来节省 NPU 显存，反向计算前从 CPU 侧搬回 NPU 来计算梯度，主要以 PCIe 带宽/延迟换取显存，通过提前预取来隐藏取回延迟。

按卸载粒度，分为两种：

- **整层 Swap（`layer_swap`）**：将指定层的激活值整体卸载到 CPU。显存收益高。
- **算子级 Swap（`op_swap`）**：仅卸载指定算子的激活值。粒度细，可灵活权衡。

> **重计算与 Swap 的本质区别**
>
> - **重计算**：丢弃前向激活值，反向再重算 —— 用 **算力** 换显存。
> - **Swap**：保留激活值但搬到 CPU 内存，反向前预取回 NPU —— 用 **PCIe 带宽/延迟** 换显存。

## 三、支持的场景与配置方法

### 选型速查

| 机制         | 配置段              | 典型场景                 | 显存收益 | 主要代价              | 关键字段                                             |
|------------|------------------|----------------------|------|-------------------|--------------------------------------------------|
| 重计算-full   | `recompute`      | 整层激活值全部丢弃，显存极紧张      | 高    | 反向重算整层前向（算力）      | `mode: full`、`full_recompute_layer`、`exclude_op` |
| 重计算-select | `recompute`      | 仅省热点模块（如 MLP），灵活权衡   | 中    | 重算选中模块（算力）        | `mode: select`、`select_module`、`exclude_op`      |
| 通信重计算      | `recompute_comm` | 切分通信算子的激活值占用较大       | 低-中  | 反向重做通信算子（算力+少量通信） | `enable`、`select_module`                         |
| 整层 Swap  | `swap`           | 重计算后仍超额，整层激活值卸载到 CPU | 高    | PCIe 带宽/延迟，通过预取隐藏 | `enable`、`layer_swap`、`default_prefetch`         |
| 算子级 Swap | `swap`           | 仅卸载指定算子的激活值          | 中    | PCIe 带宽/延迟        | `enable`、`op_swap`、`default_prefetch`            |

### 不同场景显存优化策略

大模型训练遇到显存不足(OOM)时无需同时启用所有功能。可以按牺牲性能由低到高的方式逐步降低显存占用：先用代价最低的选择重计算；若不够，再逐步扩大重计算范围，也可以叠加使用通信重计算以及Swap。

#### 第一级 · 轻度超额：选择重计算热点模块

显存仅小幅超额时，用 `select` 模式重计算 **激活值占用最大的模块**（通常是 MLP）。`select` 仅重算选中模块，算力开销远小于整层重计算，建议优先使用。

```yaml
# 假设模型 num_layers = 32
recompute:
  mode: select
  select_module:
    '.*mlp': [0-31]      # 对全部层的 mlp 模块做重计算
recompute_comm:
  enable: False
swap:
  enable: False
```

#### 第二级 · 中度超额：完全重计算指定层区间

当 `select` 仍不足以缓解显存压力时，对部分层做 **整层重计算**。单层收益最大，通常只需覆盖前若干层即可显著缓解。

```yaml
# 假设模型 num_layers = 32
recompute:
  mode: full
  full_recompute_layer: [0-15]   # 前 16 层整层重计算
recompute_comm:
  enable: False
swap:
  enable: False
```

> 启用张量并行或专家并行时，整层重计算会在反向时重复发起通信算子（如 AllGather、ReduceScatter）。可通过 `exclude_op` 排除这些通信算子，保留其前向输出以避免重复通信：

```yaml
# full 重计算 + exclude_op 排除通信算子
recompute:
  mode: full
  full_recompute_layer: [0-15]
  exclude_op:
    '.*allgather': [0-15]       # 排除 AllGather
    '.*reducescatter': [0-15]   # 排除 ReduceScatter
    '.*alltoall': [0-15]        # 排除 AllToAll
    '.*alltoallsingle': [0-15]  # 排除 AllToAllSingle
recompute_comm:
  enable: False
swap:
  enable: False
```

#### 第三级 · 重度超额：完全重计算 或 重计算 + 激活 Swap

可对全部层做 **完全重计算**（`full_recompute_layer` 覆盖所有层），以最大算力代价换取最大显存节省。若完全重计算仍无法满足，或算力不足以支撑完全重计算，可引入 **激活 Swap**：将另一批层的激活值卸载到 CPU，与重计算分担不同层。

> **同一层不能同时做重计算与 Swap**，否则配置校验报错。下例中 `0-15` 走重计算、`16-30` 走 Swap，互不重叠。

```yaml
# 假设模型 num_layers = 32
recompute:
  mode: full
  full_recompute_layer: [0-31]   # 全部层整层重计算（完全重计算）
recompute_comm:
  enable: False
swap:
  enable: False
```

```yaml
# 假设模型 num_layers = 32，重计算 + Swap 组合
recompute:
  mode: full
  full_recompute_layer: [0-15]   # 前 16 层重计算
recompute_comm:
  enable: False
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [16-30]            # 16-30 层整层激活 Swap，与重计算层不重叠
```

### 重计算（recompute）配置

| 参数名称                   | 数据类型        | 是否可选 | 默认值      | 取值说明                                                                                                                                                                                                                                         |
|------------------------|-------------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                 | str         | 可选   | `"None"` | 重计算模式：`None` / `full` / `select`。                                                                                                                                                                                                |
| `full_recompute_layer` | list/tuple  | 可选   | `None`   | 整层重计算的层范围，如 `[0-3, 8]`。`mode=full` 时必填，`mode=select` 时也可指定。                                                                                                                                  |
| `select_module`        | dict        | 可选   | `None`   | `select` 模式下的「模块路径正则 → 层范围」映射，如 `{'.*mlp': [0-31]}`。`mode=select` 时必填。                                                                                                                                                                                      |
| `exclude_op`           | dict        | 可选   | `None`   | 重计算时需排除的模块/算子，格式与 `select_module` 一致。匹配到的模块/算子在重算时直接复用前向输出，跳过重算。对 `full` 和 `select` 模式均生效，`mode=None` 时无效。 |

> **层范围格式**：单层写 `5`，区间写 `0-19`；同一列表内层号须升序且不重叠；层号不得超出 `[0, num_layers-1]`。

**选择重计算模块与算子**：`select_module` 的键为层内模块/算子路径的正则，既可匹配子模块，也可匹配算子（可调用属性）：

```yaml
# 假设模型 num_layers = 8
recompute:
  mode: select
  select_module:
    '.*attention': [0-3]          # 0-3 层重计算 attention 子模块
    '.*mlp': [4-7]                # 4-7 层重计算 mlp 子模块
    'self_attention.cast': [5]    # 5 层 attention 的 cast 算子也做重计算
recompute_comm:
  enable: False
swap:
  enable: False
```

`exclude_op` 支持排除三类目标，可在同一个字典中组合使用：

**排除指定模块（cell）**：整个子模块在重算时直接复用前向缓存输出，跳过该模块的所有计算。

```yaml
# 假设模型 num_layers = 8
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    'self_attention.linear_proj': [0-1]   # 0-1 层排除 attention 输出投影模块
    'mlp.linear_fc2': [2-3]               # 2-3 层排除 MLP 输出投影模块
recompute_comm:
  enable: False
swap:
  enable: False
```

**排除指定普通算子（可调用属性）**：模块上的函数或算子属性（如 `add`、`reshape`、`cast`、`sigmoid` 等）在重算时保留前向输出。

```yaml
# 假设模型 num_layers = 8
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    '.*add': [2-3]              # 全局匹配名为 add 的算子属性
    'self_attention.cast': [5]  # 5 层 attention 的 cast
recompute_comm:
  enable: False
swap:
  enable: False
```

**排除指定通信算子**：TP / EP 通信算子在重算时保留前向输出，避免反向重复发起集合通信。

```yaml
# 假设模型 num_layers = 8，开启 TP + EP
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    '.*allgather': [4-7]                                      # 排除 AllGather
    'self_attention.linear_proj.output.reducescatter': [4-7]  # 排除指定位置的 ReduceScatter
recompute_comm:
  enable: False
swap:
  enable: False
```

### 通信重计算（recompute_comm）配置

| 参数名称             | 数据类型  | 是否可选 | 默认值     | 取值说明                                 |
|------------------|-------|------|---------|--------------------------------------|
| `enable`         | bool  | 可选   | `False` | 是否启用通信重计算。                           |
| `select_module`  | dict  | 可选   | `None`  | 「通信算子路径 → 层范围」映射；`enable=True` 时必填。  |

**对 all-gather 通信算子重计算**

```yaml
# 假设模型 num_layers = 8
recompute_comm:
  enable: True
  select_module:
    '.*\.all_gather': [0-3]   # 0-3 层的 all_gather 算子做通信重计算
recompute:
  mode: None
swap:
  enable: False
```

### 激活值 Swap 配置

| 参数名称               | 数据类型  | 是否可选 | 默认值     | 取值说明                                                              |
|--------------------|-------|------|---------|-------------------------------------------------------------------|
| `enable`           | bool  | 可选   | `False` | 是否启用激活值 Swap。                                                     |
| `default_prefetch` | int   | 可选   | `1`     | 反向计算时提前预取前方第 N 层的激活值回 NPU，用于隐藏 CPU→NPU 取回延迟。 |
| `layer_swap`       | list  | 可选   | `None`  | 整层 Swap 条目列表，每项为 `{layers: [...]}`。                               |
| `op_swap`          | list  | 可选   | `None`  | 算子级 Swap 条目列表，每项为 `{op_name: ..., layers: [...]}`。                |

> **Swap 目前不支持流水线并行**：当 `pp > 1` 时，启用 Swap 会在配置校验阶段直接拦截。如需在流水线并行场景下节省显存，请使用重计算。

**场景：整层 Swap + 算子级 Swap**

```yaml
# 假设模型 num_layers = 8
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [0-1]        # 前 2 层整层 Swap
  op_swap:
    - op_name: '.*mlp'
      layers: [2-3]        # 2-3 层的 mlp 算子做 Swap
recompute:
  mode: None
recompute_comm:
  enable: False
```

### 组合约束

- **同层互斥**：同一层不能同时配置整层重计算与整层 Swap；算子级重计算与算子级 Swap 也不允许落在同一模块上，否则校验报错。规划时须确保重计算层与 Swap 层不重叠。
- `recompute.mode` 不为 None 时必须提供对应的 `full_recompute_layer`（full）或 `select_module`（select）；`recompute_comm.enable` 为 True 时必须提供 `select_module`。

## 相关文档

- 整体训练流程与上手：[训练指南](../guide/training.md)
- 快速上手示例：[快速开始](../quick_start/quick_start.md)
- 配置文件总览与字段上下文：[配置文件说明](./configuration.md)
- 数据侧节省显存（压缩 EOD mask、变长 FlashAttention）：[数据集](./dataset.md)
- 框架能力总览：[概述](../introduction/overview.md)
