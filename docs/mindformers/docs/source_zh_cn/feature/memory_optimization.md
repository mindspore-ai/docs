# 训练内存优化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_zh_cn/feature/memory_optimization.md)

大模型训练中，**激活值（activation）** 通常是显存占用的主要来源。MindSpore Transformers 动态图（PyNative）提供多种显存优化功能，可在配置文件中独立或组合启用，核心思路是以 **算力** 或 **PCIe 带宽** 换取显存。

所有功能由 `mindformers/pynative/distributed/activation_checkpoint.py` 的 `apply_ac` 统一使能。配置分别映射到 `RecomputeConfig` / `RecomputeCommConfig` / `SwapConfig`，详见 `mindformers/pynative/config/config.py`。

本文首先提供选型速查表，以便快速选择合适的优化方式；随后按显存压力由轻到重给出场景化推荐；最后对每种优化机制分别介绍其原理、适用场景和配置方法，并给出完整的 YAML 配置示例。

## 选型速查

| 机制         | 配置段              | 典型场景                 | 显存收益 | 主要代价              | 关键字段                                             |
|------------|------------------|----------------------|------|-------------------|--------------------------------------------------|
| 重计算-full   | `recompute`      | 整层激活值全部丢弃，显存极紧张      | 高    | 反向重算整层前向（算力）      | `mode: full`、`full_recompute_layer`、`exclude_op` |
| 重计算-select | `recompute`      | 仅省热点模块（如 MLP），灵活权衡   | 中    | 重算选中模块（算力）        | `mode: select`、`select_module`、`exclude_op`      |
| 通信重计算      | `recompute_comm` | 切分通信算子的激活值占用较大       | 低-中  | 反向重做通信算子（算力+少量通信） | `enable`、`select_module`                         |
| SWAP-layer | `swap`           | 重计算后仍超额，整层激活值卸载到 CPU | 高    | PCIe 带宽/延迟，通过预取隐藏 | `enable`、`layer_swap`、`default_prefetch`         |
| SWAP-op    | `swap`           | 仅卸载指定算子的激活值          | 中    | PCIe 带宽/延迟        | `enable`、`op_swap`、`default_prefetch`            |

> **两者的本质区别**
>
> - **重计算**：丢弃前向激活值，反向再重算 —— 用 **算力** 换显存。
> - **SWAP**：保留激活值但搬到 CPU 内存，反向前预取回 NPU —— 用 **PCIe 带宽/延迟** 换显存。

## 显存不够时怎么选（场景化策略）

遇到 OOM 时无需同时启用所有功能。重计算与 SWAP 在减少激活值显存方面效果相近，区别在于代价不同：重计算以 **算力** 换显存，SWAP 以 **PCIe 带宽/延迟** 换显存。可根据集群资源余量选择合适的方式，也可组合使用（分配到不同层）。以下按代价由低到高给出推荐路径：先用代价最低的选择重计算；若不够，再扩大重计算范围；仍不够，再组合 SWAP。

### 第一级 · 轻度超额：选择重计算热点模块

显存仅小幅超额时，先用 `select` 模式选择重计算 **激活值占用最大的模块**（通常是 MLP）。`select` 仅重算选中模块的前向，算力开销远小于整层重计算。

> 重计算以算力换显存。`select` 粒度细、开销小，建议优先使用。

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

### 第二级 · 中度超额：完全重计算指定层区间

当 `select` 仍不足以缓解显存压力时，对部分层做 **整层重计算**。整层激活值全部丢弃、反向重算，单层收益最大；通常只需覆盖前若干层即可显著缓解。

> 完全重计算模式节省显存最多，但算力开销也最大。建议仅对必要的层启用，按需扩大 `full_recompute_layer` 区间。

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
  exclude_op: ["AllGather", "ReduceScatter", "AllToAll"]
recompute_comm:
  enable: False
swap:
  enable: False
```

### 第三级 · 重度超额：完全重计算 或 重计算 + 激活 SWAP

选择重计算仍不够时，可对全部层做 **完全重计算**（`full_recompute_layer` 覆盖所有层），以最大算力代价换取最大显存节省。若完全重计算仍无法满足，或算力不足以支撑完全重计算，可引入 **激活 SWAP**：将另一批层的激活值卸载到 CPU 内存，反向前预取回 NPU，与重计算分担不同层。

> - SWAP 以 PCIe 带宽/延迟换显存，取回延迟通过 `default_prefetch` 提前预取来隐藏。
> - **同一层不能同时做重计算与 SWAP**，否则配置校验报错。下例中 `0-15` 走重计算、`16-30` 走 SWAP，互不重叠。

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
# 假设模型 num_layers = 32，重计算 + SWAP 组合
recompute:
  mode: full
  full_recompute_layer: [0-15]   # 前 16 层重计算
recompute_comm:
  enable: False
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [16-30]            # 16-30 层整层激活 SWAP，与重计算层不重叠
```

## 重计算（recompute）

### 概述

重计算（Activation Checkpointing）可以显著降低训练时的激活内存占用，但会额外增加一些计算开销。其核心思想是在前向传播阶段丢弃部分中间激活值，在反向传播时再重新计算所需激活值，以算力换取显存。关于重计算的原理和框架侧能力可参考 [MindSpore 教程文档：重计算](https://www.mindspore.cn/tutorials/zh-CN/r2.10.0/parallel/recompute.html)。

动态图模式下，`recompute` 段通过 `mode` 字段控制重计算的粒度：

- `None`：关闭重计算。
- `full`：完全重计算，对 `full_recompute_layer` 指定的整层进行重计算。
- `select`：选择重计算，对 `select_module` 指定的模块或算子做选择性重计算，可对 `full_recompute_layer` 指定的层同时做整层重计算。

### 适用场景

- 显存超额较小、需精确控制重计算范围时，使用 `select`（如只重算 `.*mlp`）。
- 显存超额较大、需要最大显存节省时，使用 `full`，并仅覆盖必要的层区间。

### 字段说明

| 参数名称                   | 数据类型        | 是否可选 | 默认值      | 取值说明                                                                                                                                                                                                                                         |
|------------------------|-------------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                 | str         | 可选   | `"None"` | 重计算模式：`None` / `full` / `select`（其它取值在构造时报错）。                                                                                                                                                                                                |
| `full_recompute_layer` | list/tuple  | 可选   | `None`   | 整层重计算的层范围，元素为层号或区间，如 `[0-3, 8]`。`mode=full` 时必填，`mode=select` 时也可指定，指定层做整层重计算、其余层按 `select_module` 做选择性重计算。                                                                                                                                  |
| `select_module`        | dict        | 可选   | `None`   | `select` 模式下的「模块路径 → 层范围」映射，键为模块路径正则。`mode=select` 时必填。                                                                                                                                                                                      |
| `exclude_op`           | list/tuple  | 可选   | `None`   | 重计算时需排除的算子名称列表。当 `mode` 为 `full` 或 `select` 时，若某个算子的名称包含列表中某项（大小写不敏感），则该算子的前向输出会被保留（`MUST_SAVE`），反向时直接复用而非重算；其余算子仍正常重计算。例如 `["AllGather", "ReduceScatter", "AllToAll"]` 保留通信算子输出以避免反向重算时重复发起集合通信。匹配基于算子名称的全局子串匹配，不限定于特定模块路径。`mode=None` 时无效。 |

### 选择重计算的关键行为

`select_module` 的键不是固定字段名，而是 **层内模块/算子路径的正则**：

- 框架先用 `_get_single_layer_whitelist` 收集每层的全部子模块（cell）与算子（function）路径，构成白名单。
- 再用 `regex.fullmatch` 对白名单中每条路径做 **全匹配**（不是部分匹配），匹配成功后对该模块/算子使能重计算。因此正则需匹配 **整条路径**，例如 `.*mlp` 可匹配名为 `mlp` 的子模块。
- **父模块配置自动覆盖子模块**：若某层已配置父模块（如 `attention`），其子模块（如 `attention.core`）会被去重跳过，无需重复列出。
- **未匹配到任何模块只告警、不报错**：正则写错或层内无对应模块时，日志输出 `select_module pattern '...' did not match any module`，训练继续，但该项不生效。

> **层范围校验**：`select_module` 每个键的 `ranges` 与 `full_recompute_layer` 使用相同的校验规则（`_validate_layer_specs`）：
>
> - 单层写 `5`，区间写 `0-19`（`start <= end`）。
> - 同一列表内层号须 **严格升序且不重叠**。
> - 层号不得超出模型层数 `[0, num_layers-1]`。

### exclude_op 的关键行为

`exclude_op` 用于在重计算时排除特定算子，使其前向输出被保留而非丢弃：

- 匹配方式为 **大小写不敏感的子串匹配**：若算子名称包含列表中某项，则该算子输出保留。例如 `["AllGather"]` 可匹配 `InnerCommAllGather`。
- 匹配范围是 **全局的**，不限定于特定模块路径。例如 `["AllGather"]` 会保留重计算 cell 内所有名称包含 `AllGather` 的算子输出（如 `InnerCommAllGather`）。
- 该字段对 `full` 和 `select` 模式均生效，`mode=None` 时无效。
- 使用 `exclude_op` 保留算子输出会额外占用显存（保留的张量不再被丢弃），需在显存与重计算算力之间权衡。

### 场景化配置：选择重计算 attention 与 MLP

```yaml
# 假设模型 num_layers = 8
recompute:
  mode: select
  select_module:
    '.*attention': [0-3]   # 0-3 层重计算 attention 子模块
    '.*mlp': [4-7]         # 4-7 层重计算 mlp 子模块
recompute_comm:
  enable: False
swap:
  enable: False
```

## 通信重计算（recompute_comm）

### 概述

`recompute_comm` 对选定的 **通信算子** 做重计算。其 `enable` 与 `recompute.mode` **相互独立**，可单独启用，也可与重计算并存。

### 适用场景

并行切分引入的通信算子（如 all-gather / reduce-scatter）的激活值占用较大、又不想整层重计算时，单独对这些通信算子重计算。

### 字段说明

| 参数名称             | 数据类型  | 是否可选 | 默认值     | 取值说明                                 |
|------------------|-------|------|---------|--------------------------------------|
| `enable`         | bool  | 可选   | `False` | 是否启用通信重计算。                           |
| `select_module`  | dict  | 可选   | `None`  | 「通信算子路径 → 层范围」映射；`enable=True` 时必填。  |

> 通信重计算的 `select_module` 必须匹配到 **算子**。若正则命中的是一个 cell（子模块）而非算子，日志会提示「is expected to be operation but got cell, this configuration will not be effective」，该项 **不生效**。匹配、层范围校验、父子去重、未匹配告警等行为与重计算 `select` 一致。

### 场景化配置：对 all-gather 通信算子重计算

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

## 激活值 SWAP（swap）

### 概述

`swap` 段把激活值卸载到 CPU 内存，反向计算前再预取回 NPU，以 PCIe 带宽/延迟换取显存。支持 **整层卸载（`layer_swap`）** 与 **算子级卸载（`op_swap`）**。框架通过策略函数自动跳过注意力掩码等需常驻 NPU 的张量。

> **SWAP 目前不支持流水线并行**：当 `pp > 1` 时，启用 SWAP 会在配置校验阶段直接拦截。如需在流水线并行场景下节省显存，请使用重计算。

### 适用场景

当重计算无法进一步释放显存时，对未做重计算的层启用 SWAP。取回延迟通过 `default_prefetch` 在反向 FlashAttention 算子前提前预取来隐藏。

### 字段说明

| 参数名称               | 数据类型  | 是否可选 | 默认值     | 取值说明                                                              |
|--------------------|-------|------|---------|-------------------------------------------------------------------|
| `enable`           | bool  | 可选   | `False` | 是否启用激活值 SWAP。                                                     |
| `default_prefetch` | int   | 可选   | `1`     | 反向计算时预取激活值的层偏移量，即在反向计算当前层时提前预取前方第 N 层的激活值回 NPU，用于隐藏 CPU→NPU 取回延迟。 |
| `layer_swap`       | list  | 可选   | `None`  | 整层 SWAP 条目列表，每项为 `{layers: [...]}`。                               |
| `op_swap`          | list  | 可选   | `None`  | 算子级 SWAP 条目列表，每项为 `{op_name: ..., layers: [...]}`。                |

> - **`default_prefetch`** **必须落在** **`[1, num_layers-1]`**，否则校验报错。
> - **最大层号 +** **`default_prefetch`** **不得达到** **`num_layers`**（即需 `max_layer + prefetch < num_layers`），否则预取会越界报错；最大可用层号为 `num_layers - prefetch - 1`。
> - **`layer_swap`** **实际只取第一条 entry**（源码 `sc.layer_swap[0]`）。配置多条 `layer_swap` 时，第二条及以后会被忽略，所有需整层 SWAP 的层应合并写入第一条的 `layers` 列表。
> - `op_swap` 的层范围同样须满足升序、不越界校验。

### 场景化配置：整层 SWAP + 算子级 SWAP

```yaml
# 假设模型 num_layers = 8
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [0-1]        # 前 2 层整层 SWAP
  op_swap:
    - op_name: '.*mlp'
      layers: [2-3]        # 2-3 层的 mlp 算子做 SWAP
recompute:
  mode: None
recompute_comm:
  enable: False
```

## 组合约束

> **同层互斥**：同一层不能同时配置整层重计算与整层 SWAP；算子级重计算与算子级 SWAP 也不允许落在同一模块（含父子模块）上，否则在 `apply_ac` 的校验阶段（`_check_recompute_swap_overlap`）直接报错。规划时须确保重计算层与 SWAP 层不重叠。

- `recompute.mode != "None"` 时必须提供对应的 `full_recompute_layer`（full）或 `select_module`（select）。
- `recompute_comm.enable: True` 时必须提供 `select_module`。
- 所有层范围须 **升序、不越界**；`swap.default_prefetch` 须落在 `[1, num_layers-1]`，且最大层号加 `default_prefetch` 须小于 `num_layers`。
- 重计算与 SWAP 同时启用时，框架在使能前先做重叠检测；通过后再分别校验与使能。

## 相关文档

- 整体训练流程与上手：[训练指南](../guide/training.md)
- 快速上手示例：[快速开始](../quick_start/quick_start.md)
- 配置文件总览与字段上下文：[配置文件说明](./configuration.md)
- 数据侧节省显存（压缩 EOD mask、变长 FlashAttention）：[数据集](./dataset.md)
- 框架能力总览：[概述](../introduction/overview.md)

