# Training Memory Optimization

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/memory_optimization.md)

In foundation model training, **activation** is the main source of the graphics memory usage. The dynamic graph (PyNative) of MindSpore Transformers provides multiple graphics memory optimization functions, which can be enabled independently or in combination in the configuration file. The core idea is to use **computing power** or **PCIe bandwidth** to save graphics memory.

All functions are enabled by the `apply_ac` of `mindformers/pynative/distributed/activation_checkpoint.py`. The configurations are mapped to `RecomputeConfig`, `RecomputeCommConfig`, or `SwapConfig`. For details, see `mindformers/pynative/config/config.py`.

This document first provides a quick reference table for selecting the appropriate optimization method. Then, it provides scenario-based recommendations based on the graphics memory pressure from light to heavy. Finally, it describes the principles, application scenarios, and configuration methods of each optimization mechanism, and provides complete YAML configuration examples.

## Quick Reference for Selection

| Mechanism        | Section             | Typical Scenario                | Graphics Memory Benefit| Major Cost             | Key Field                                            |
|------------|------------------|----------------------|------|-------------------|--------------------------------------------------|
| Recomputation - full  | `recompute`      | All activations of the entire layer are discarded, and the graphics memory is extremely insufficient.     | High   | Recomputation of the forward results of the entire layer during backpropagation (computing power)     | `mode: full`, `full_recompute_layer`, and `exclude_op`|
| Recomputation - select| `recompute`      | Only hotspot modules (such as MLP) are recomputed, and flexible trade-off is allowed.  | Medium   | Recomputation of the selected module (computing power)       | `mode: select`, `select_module`, and `exclude_op`     |
| Communication computation     | `recompute_comm` | The activations of the sharding communication operators occupies a large amount of memory.      | Low - Medium | Backward recomputation of the communication operators (computing power and a small amount of communication)| `enable` and `select_module`                        |
| SWAP-layer | `swap`           | If the activation memory usage still exceeds the limit after recomputation, the activations of the entire layer are offloaded to the CPU.| High   | PCIe bandwidth/latency, hidden by prefetching| `enable`, `layer_swap`, and `default_prefetch`        |
| SWAP-op    | `swap`           | Only the activations of the specified operators are offloaded.         | Medium   | PCIe bandwidth/latency       | `enable`, `op_swap`, and `default_prefetch`           |

> **Essential Differences Between the Two**
>
> - **Recomputation**: Discard the forward activations and recompute them during backpropagation. That is, use **computing power** to exchange for graphics memory.
> - **SWAP**: The activations are retained but moved to the CPU memory, and are prefetched back to the NPU during backpropagation. That is, use **PCIe bandwidth/latency** to exchange for graphics memory.

## How to Choose Methods When Graphics Memory Is Insufficient (Scenario-based Policies)

When OOM occurs, you do not need to enable all functions at the same time. Recomputation and SWAP have similar effects on reducing the graphics memory occupied by activations. The difference is that the cost is different. Recomputation uses **computing power** to save graphics memory, while SWAP uses **PCIe bandwidth/latency** to save graphics memory. You can select a proper method based on the remaining cluster resources, or use them together (allocated to different layers). The following provides recommended paths in ascending order of cost: First, select the recomputation path with the lowest cost. If this is not enough, expand the recomputation scope. If it is still not enough, use SWAP.

### Level 1 · Slight Overcommitment: Recomputing the Hotspot Module

If the graphics memory is slightly overcommitted, use the `select` mode to recompute the **module with activations that occupies the largest amount of memory** (usually MLP). `select` recomputes only the forward propagation of the selected module, and the computing power overhead is much lower than that of recomputing the entire layer.

> Recomputation uses computing power to save graphics memory. `select` is fine-grained and has low overhead, so it is recommended.

```yaml
# Assume that num_layers of the model is 32.
recompute:
  mode: select
  select_module:
    '.*mlp': [0-31]      # Recompute the MLP module of all layers.
recompute_comm:
  enable: False
swap:
  enable: False
```

### Level 2 · Moderate Overcommitment: Fully Recomputing a Specified Layer Range

When `select` is still insufficient to relieve the graphics memory pressure, **full recomputation** is performed on some layers. All activations of the entire layer are discarded and backpropagation is recomputed, which brings the maximum benefit to a single layer. Generally, only the first several layers need to be recomputed to significantly relieve the memory pressure.

> The full recomputation mode saves the most graphics memory but also incurs the highest computing power overhead. You are advised to enable this mode only for necessary layers and expand the `full_recompute_layer` range as required.

```yaml
# Assume that num_layers of the model is 32.
recompute:
  mode: full
  full_recompute_layer: [0-15]   # Recompute the entire first 16 layers.
recompute_comm:
  enable: False
swap:
  enable: False
```

> When tensor parallelism or expert parallelism is enabled, full-layer recomputation repeatedly initiates communication operators (such as AllGather and ReduceScatter) during the backward backpropagation. You can use `exclude_op` to exclude these communication operators and retain their forward outputs to avoid repeated communication.

```yaml
# Full recomputation and exclude_op for excluding communication operators
recompute:
  mode: full
  full_recompute_layer: [0-15]
  exclude_op: ["AllGather", "ReduceScatter", "AllToAll"]
recompute_comm:
  enable: False
swap:
  enable: False
```

### Level 3 · Severe Overcommitment: Full Recomputation or Recomputation+Activation SWAP

If the selected recomputation is still insufficient, you can perform **full recomputation** on all layers (`full_recompute_layer` covers all layers) to save the maximum graphics memory at the cost of maximum computing power. If full recomputation still cannot meet the requirements or the computing power is insufficient to support full recomputation, **activation SWAP** can be introduced. The activations of another batch of layers are offloaded to the CPU memory and prefetched back to the NPU before backpropagation, which shares different layers with recomputation.

> - SWAP uses the PCIe bandwidth/latency to save the graphics memory. The latency of data fetch is hidden by `default_prefetch` prefetching.
> - **Recomputation and SWAP cannot be performed on the same layer at the same time**. Otherwise, an error will be reported during configuration verification. In the following example, `0-15` is used for recomputation, and `16-30` is used for SWAP, and they do not overlap with each other.

```yaml
# Assume that num_layers of the model is 32.
recompute:
  mode: full
  full_recompute_layer: [0-31]   # Full recomputation on all layers
recompute_comm:
  enable: False
swap:
  enable: False
```

```yaml
# Assume that num_layers of the model is 32, and recomputation and SWAP are used together.
recompute:
  mode: full
  full_recompute_layer: [0-15]   # Recomputation on the first 16 layers
recompute_comm:
  enable: False
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [16-30]            # Activation swap for layers 16 to 30, which do not overlap with the recomputation layers
```

## Recomputation (recompute)

### Overview

Activation checkpointing can significantly reduce the activation memory usage during training, but it increases the computational overhead. The core idea is to discard some intermediate activations during the forward propagation phase and recompute the required activations during the backpropagation phase, trading off computing power for memory. For details about the principles and framework capabilities of recomputation, see [MindSpore Tutorial: Recomputation](https://www.mindspore.cn/tutorials/en/master/parallel/recompute.html).

In dynamic graph mode, the `recompute` field uses the `mode` field to control the granularity of recomputation.

- `None`: Recomputation is disabled.
- `full`: Full recomputation is performed on the layer specified by `full_recompute_layer`.
- `select`: Selective recomputation is performed on the module or operator specified by `select_module`. Full recomputation can also be performed on the layer specified by `full_recompute_layer`.

### Application Scenarios

- When the graphics memory usage slightly exceeds the limit and the recomputation range needs to be precisely controlled, use `select` (for example, recomputing only `.*mlp`).
- When the graphics memory usage greatly exceeds the limit and the maximum graphics memory needs to be saved, use `full` for only the necessary layer range.

### Field Description

| Parameter                  | Data Type       | Required/Optional| Default Value     | Value Description                                                                                                                                                                                                                                        |
|------------------------|-------------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                 | str         | Optional  | `"None"` | The recomputation mode can be set to `None`, `full`, or `select` (other values will cause an error during construction).                                                                                                                                                                                               |
| `full_recompute_layer` | list/tuple  | Optional  | `None`   | Range of layers for which full-layer recomputation is performed. The elements are layer numbers or ranges, for example, `[0-3, 8]`. This parameter is required when `mode=full` and can also be specified when `mode=select`. If this parameter is specified, full-layer recomputation is performed on the specified layers, and selective recomputation is performed on the remaining layers based on `select_module`.                                                                                                                                 |
| `select_module`        | dict        | Optional  | `None`   | Mapping from the module path to the layer range in `select` mode. The key is the regular expression of the module path. This parameter is required when `mode=select`.                                                                                                                                                                                     |
| `exclude_op`           | list/tuple  | Optional  | `None`   | List of operator names to be excluded during recomputation. When `mode` is set to `full` or `select`, if the name of an operator contains an item in the list (case-insensitive), the forward output of the operator is retained (`MUST_SAVE`), and the output is directly reused instead of being recomputed during the backpropagation. Other operators are still recomputed normally. For example, `["AllGather", "ReduceScatter", "AllToAll"]` retains the output of the communication operators to avoid repeated collective communication during backpropagation. The global substring matching is performed based on the operator name, and is not limited to a specific module path. This parameter is invalid when `mode=None`.|

### Selecting Key Behaviors for Recomputation

The key of `select_module` is not a fixed field name, but the **regular expression of the module/operator path within the layer**.

- The framework uses `_get_single_layer_whitelist` to collect the paths of all child modules (cells) and operators (functions) at each layer to form a whitelist.
- Then, `regex.fullmatch` is used to perform **full matching** (not partial matching) on each path in the whitelist. After the matching is successful, recomputation is enabled for the cell/function. Therefore, the regular expression must match the **entire path**. For example, `.*mlp` can match the cell named `mlp`.
- **Automatic overwriting of child modules by parent modules**: If a parent module (such as `attention`) has been configured at a layer, its child module (such as `attention.core`) will be skipped and do not need to be listed again.
- **Only an alarm is reported if no module is matched**: If the regular expression is incorrect or there is no corresponding module in the layer, `select_module pattern '...' did not match any module` is output in the log, and the training continues, but this item does not take effect.

> **Layer range verification**: `ranges` and `full_recompute_layer` of each key in `select_module` use the same verification rule (`_validate_layer_specs`):
>
> - Write `5` for a single layer and `0-19` (`start <= end`) for a range of layers.
> - The layer numbers in the same list must be **in ascending order and cannot overlap**.
> - The layer number cannot exceed the number of model layers (`[0, num_layers-1]`).

### Key Behaviors of exclude_op

`exclude_op` is used to exclude specific operators during recomputation, so that the forward output of the operators is retained instead of being discarded.

- The matching mode is **case-insensitive substring matching**. If the operator name contains an item in the list, the output of the operator is retained. For example, `["AllGather"]` can match `InnerCommAllGather`.
- The matching scope is **global** and is not limited to a specific module path. For example, `["AllGather"]` retains the outputs of all operators whose names contain `AllGather` (for example, `InnerCommAllGather`) in the recomputation cell.
- This field takes effect for both the `full` and `select` modes and is invalid when `mode=None`.
- Using `exclude_op` to retain operator outputs will occupy additional graphics memory (the retained tensors will no longer be discarded). Therefore, a trade-off between graphics memory and recomputation computing power is required.

### Scenario-based Configuration: Selective Recomputation of attention and MLP

```yaml
# Assume that num_layers of the model is 8.
recompute:
  mode: select
  select_module:
    '.*attention': [0-3]   # Recompute the attention cell at layers 0 to 3.
    '.*mlp': [4-7]         # Recompute the MLP cell at layers 4 to 7.
recompute_comm:
  enable: False
swap:
  enable: False
```

## Communication Recomputation (recompute_comm)

### Overview

`recompute_comm` performs recomputation on the selected **communication operators**. Its `enable` and `recompute.mode` are **independent of each other** and can be enabled separately or together with recomputation.

### Application Scenarios

When the activations of communication operators (such as AllGather and ReduceScatter) introduced by parallel splitting occupy a large amount of memory and you do not want to recompute the entire layer, you can recompute these communication operators separately.

### Field Description

| Parameter            | Data Type | Required/Optional| Default Value    | Value Description                                |
|------------------|-------|------|---------|--------------------------------------|
| `enable`         | bool  | Optional  | `False` | Specifies whether to enable communication recomputation.                          |
| `select_module`  | dict  | Optional  | `None`  | Specifies the mapping from the communication operator path to the layer range. This parameter is required under the condition of `enable=True`. |

> `select_module` for communication recomputation must match an **operator**. If the regular expression matches a cell instead of a function, the log will display the message "is expected to be operation but got cell, this configuration will not be effective". In this case, this item **does not take effect**. The behaviors such as matching, layer range verification, parent-child deduplication, and unmatched alarms are the same as those of recomputation in `select` mode.

### Scenario-based Configuration: Recomputing the all-gather Communication Operator

```yaml
# Assume that num_layers of the model is 8.
recompute_comm:
  enable: True
  select_module:
    '.*\.all_gather': [0-3] # AllGather operators at layers 0 to 3 are used for communication recomputation.
recompute:
  mode: None
swap:
  enable: False
```

## Activation SWAP (swap)

### Overview

The `swap` section offloads activations to the CPU memory and prefetches them back to the NPU before backward computation, trading the PCIe bandwidth/latency for the graphics memory. Both **full-layer offloading (`layer_swap`)** and **operator-level offloading (`op_swap`)** are supported. The framework automatically skips tensors that need to reside on the NPU, such as the attention mask, through the policy function.

> **SWAP does not support pipeline parallelism**. When `pp > 1`, enabling SWAP will be directly intercepted in the configuration verification phase. If you want to save the graphics memory in the pipeline parallelism scenario, use recomputation.

### Application Scenarios

When recomputation cannot further release the graphics memory, enable SWAP for layers that are not recomputed. The fetch latency is hidden by using `default_prefetch` to prefetch the data before the FlashAttention operator backpropagation.

### Field Description

| Parameter              | Data Type | Required/Optional| Default Value    | Value Description                                                             |
|--------------------|-------|------|---------|-------------------------------------------------------------------|
| `enable`           | bool  | Optional  | `False` | Specifies whether to enable activation SWAP.                                                    |
| `default_prefetch` | int   | Optional  | `1`     | Layer offset for prefetching activations during backward computation. That is, during backward computation of the current layer, the activations of the previous *N*th layer are prefetched to the NPU in advance to hide the latency of fetching data from the CPU to the NPU.|
| `layer_swap`       | list  | Optional  | `None`  | List of SWAP items at the entire layer. Each item is `{layers: [...]}`.                              |
| `op_swap`          | list  | Optional  | `None`  | List of operator-level SWAP items. Each item is `{op_name: ..., layers: [...]}`.               |

> - The value of **`default_prefetch`** **must be within** **`[1, num_layers-1]`**. Otherwise, an error is reported during verification.
> - The sum of **`maximum layer number`** and **`default_prefetch`** **must be less than** **`num_layers`** (that is, **max_layer** + **prefetch** < **num_layers**). Otherwise, an out-of-bounds error is reported.
> - **`layer_swap`** **actually takes only the first entry** (`sc.layer_swap[0]` in the source code). When multiple `layer_swap` configurations are provided, the second and subsequent configurations are ignored. All layers that require full-layer SWAP should be combined and written into the `layers` list of the first configuration.
> - The layer range of `op_swap` must also meet the ascending order and out-of-bounds verification requirements.

### Scenario-based Configuration: Full-Layer SWAP + Operator-Level SWAP

```yaml
# Assume that num_layers of the model is 8.
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [0-1]        # Full-layer SWAP for the first two layers
  op_swap:
    - op_name: '.*mlp'
      layers: [2-3]        #SWAP for the MLP operators at layers 2 and 3
recompute:
  mode: None
recompute_comm:
  enable: False
```

## Combination Constraints

> **Intra-layer exclusion**: Full-layer recomputation and full-layer SWAP cannot be configured for the same layer. Operator-level recomputation and operator-level SWAP cannot be configured for the same module (including parent and child modules). Otherwise, an error will be reported during the `apply_ac` verification phase (`_check_recompute_swap_overlap`). During planning, ensure that the recomputation layer and the SWAP layer do not overlap.

- The corresponding `full_recompute_layer` (full) or `select_module` (select) must be provided under the condition of `recompute.mode != "None"`.
- The `select_module` must be provided under the condition of `recompute_comm.enable: True`.
- The layer range must be in ascending order and **cannot be out of bounds**. `swap.default_prefetch` must be within the range of `[1, num_layers-1]`, and the maximum layer ID plus `default_prefetch` must be less than `num_layers`.
- When recomputation and SWAP are enabled at the same time, the framework performs overlapping detection before enabling them. After the detection is passed, the framework verifies and enables them separately.

## Related Documents

- Overall training process and quick start: [Training Guide](../guide/training.md)
- Quick start example: [Quick Start](../quick_start/quick_start.md)
- Configuration file overview and field context: [Configuration File Description](./configuration.md)
- Reducing the graphics memory usage on the data side (EOD mask compression and variable-length FlashAttention): [Datasets](./dataset.md)
- Framework capability overview: [Overview](../introduction/overview.md)
