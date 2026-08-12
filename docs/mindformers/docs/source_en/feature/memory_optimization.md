# Training Graphics Memory Optimization

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/memory_optimization.md)

In foundation model training, **activation** is the main source of the graphics memory usage. The dynamic graph (PyNative) of MindSpore Transformers provides multiple graphics memory optimization functions, which can be enabled independently or in combination in the configuration file. This document introduces the concept and necessity of graphics memory optimization, the principles of various methods, and the supported scenarios and configuration methods of MindSpore Transformers.

## I. Concept and Necessity of Graphics Memory Optimization

The graphics memory usage of foundation model training mainly comes from four parts: **model weights**, **optimizer state**, **gradients**, and **activations**. The first three can be distributed to multiple cards through parallel splitting (TP/PP/EP), while activations are strongly related to the sequence length, batch, and number of layers, and **are usually the graphics memory bottleneck in foundation model training**.

Therefore, the core goal of graphics memory optimization is to reduce the graphics memory usage of activations without changing the numerical results. The essence is to **trade other resources for graphics memory**, mainly in two ways:

- **Trade computing power for graphics memory**: Discard forward activations and recompute them during backpropagation — that is, **Recompute**.
- **Trade PCIe bandwidth/latency for graphics memory**: Retain activations but move them to the CPU memory and prefetch them back to the NPU before backpropagation — that is, **Activation Swap**.

The two have similar effects, but the costs are different: recomputation consumes extra computing power, while Swap consumes PCIe bandwidth and latency. You can choose a suitable method based on the remaining cluster resources, or use them together (allocated to different layers).

## II. Graphics Memory Optimization Methods

### Recompute

[Recomputation](https://www.mindspore.cn/tutorials/en/master/parallel/recompute.html) (Activation Checkpointing) discards some intermediate activations during forward propagation and recomputes the required activations during backpropagation, trading computing power for graphics memory.

Based on the granularity of discarding and recomputation, it is divided into two categories:

- **Activation recomputation**: Performs recomputation on model layers or modules within a layer.
  - `full` (full recomputation): Performs recomputation on the entire specified layer, and all activations of the entire layer are discarded. The graphics memory benefit is the largest, and the computing power overhead is also the largest.
  - `select` (selective recomputation): Performs recomputation only on specified modules (such as MLP). Fine-grained and low overhead, suitable for precisely controlling the scope.
- **Communication recomputation**: When the activations of communication operators (such as AllGather and ReduceScatter) introduced by parallel splitting (TP/EP) occupy a large amount of graphics memory, recomputation is performed separately on these communication operators without recomputing the entire layer.

> Communication recomputation is independent of activation recomputation and can be enabled or disabled at the same time.

### Activation Swap

Activation Swap saves NPU graphics memory by offloading activations to the CPU memory. Before backward computation, the activations are moved back to the NPU from the CPU side to compute gradients. It primarily trades PCIe bandwidth/latency for graphics memory, and hides the fetch latency through prefetching.

Based on the offloading granularity, it is divided into two types:

- **Full-layer Swap (`layer_swap`)**: Offloads the activations of the specified layer to the CPU as a whole. High graphics memory benefit.
- **Operator-level Swap (`op_swap`)**: Offloads only the activations of the specified operator. Fine-grained and flexible.

> **Essential Differences Between Recomputation and Swap**
>
> - **Recomputation**: Discard the forward activations and recompute them during backpropagation — use **computing power** to exchange for graphics memory.
> - **Swap**: Retain the activations but move them to the CPU memory and prefetch them back to the NPU before backpropagation — use **PCIe bandwidth/latency** to exchange for graphics memory.

## III. Supported Scenarios and Configuration Methods

### Quick Reference for Selection

| Mechanism        | Section             | Typical Scenario                | Graphics Memory Benefit| Major Cost             | Key Field                                            |
|------------|------------------|----------------------|------|-------------------|--------------------------------------------------|
| Recomputation - full  | `recompute`      | All activations of the entire layer are discarded, and the graphics memory is extremely insufficient.     | High   | Recomputation of the forward results of the entire layer during backpropagation (computing power)     | `mode: full`, `full_recompute_layer`, and `exclude_op`|
| Recomputation - select| `recompute`      | Only hotspot modules (such as MLP) are recomputed, and flexible trade-off is allowed.  | Medium   | Recomputation of the selected module (computing power)       | `mode: select`, `select_module`, and `exclude_op`     |
| Communication recomputation     | `recompute_comm` | The activations of the sharding communication operators occupy a large amount of graphics memory.      | Low - Medium | Backward recomputation of the communication operators (computing power and a small amount of communication)| `enable` and `select_module`                        |
| Full-layer Swap  | `swap`           | If the graphics memory usage still exceeds the limit after recomputation, the activations of the entire layer are offloaded to the CPU.| High   | PCIe bandwidth/latency, hidden by prefetching| `enable`, `layer_swap`, and `default_prefetch`        |
| Operator-level Swap | `swap`           | Only the activations of the specified operators are offloaded.         | Medium   | PCIe bandwidth/latency       | `enable`, `op_swap`, and `default_prefetch`           |

### Graphics Memory Optimization Strategies for Different Scenarios

When foundation model training encounters insufficient graphics memory (OOM), you do not need to enable all functions at the same time. You can gradually reduce the graphics memory usage by sacrificing performance from low to high: First, use selective recomputation with the lowest cost. If this is not enough, gradually expand the recomputation scope, and you can also use communication recomputation and Swap in combination.

#### Level 1 · Slight Overcommitment: Selectively Recomputing the Hotspot Module

If the graphics memory is slightly overcommitted, use the `select` mode to recompute the **module with activations that occupies the largest amount of graphics memory** (usually MLP). `select` recomputes only the selected module, and the computing power overhead is much lower than that of recomputing the entire layer. It is recommended to use it first.

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

#### Level 2 · Moderate Overcommitment: Fully Recomputing a Specified Layer Range

When `select` is still insufficient to relieve the graphics memory pressure, perform **full recomputation** on some layers. The single-layer benefit is the largest, and generally only the first several layers need to be covered to significantly relieve the pressure.

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

> When tensor parallelism or expert parallelism is enabled, full-layer recomputation repeatedly initiates communication operators (such as AllGather and ReduceScatter) during backpropagation. You can use `exclude_op` to exclude these communication operators and retain their forward outputs to avoid repeated communication:

```yaml
# Full recomputation + exclude_op to exclude communication operators
recompute:
  mode: full
  full_recompute_layer: [0-15]
  exclude_op:
    '.*allgather': [0-15]       # Exclude AllGather
    '.*reducescatter': [0-15]   # Exclude ReduceScatter
    '.*alltoall': [0-15]        # Exclude AllToAll
    '.*alltoallsingle': [0-15]  # Exclude AllToAllSingle
recompute_comm:
  enable: False
swap:
  enable: False
```

#### Level 3 · Severe Overcommitment: Full Recomputation or Recomputation + Activation Swap

You can perform **full recomputation** on all layers (`full_recompute_layer` covers all layers) to save the maximum graphics memory at the cost of maximum computing power. If full recomputation still cannot meet the requirements or the computing power is insufficient to support full recomputation, **activation Swap** can be introduced: offload the activations of another batch of layers to the CPU to share different layers with recomputation.

> **Recomputation and Swap cannot be performed on the same layer at the same time**. Otherwise, an error is reported during configuration verification. In the following example, `0-15` uses recomputation and `16-30` uses Swap, and they do not overlap.

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
# Assume that num_layers of the model is 32, and recomputation and Swap are used together.
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

### Recomputation (recompute) Configuration

| Parameter                  | Data Type       | Required/Optional| Default Value     | Value Description                                                                                                                                                                                                                                        |
|------------------------|-------------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                 | str         | Optional  | `"None"` | Recomputation mode: `None` / `full` / `select`.                                                                                                                                                                                               |
| `full_recompute_layer` | list/tuple  | Optional  | `None`   | Range of layers for which full-layer recomputation is performed, for example, `[0-3, 8]`. This parameter is required when `mode=full` and can also be specified when `mode=select`.                                                                  |
| `select_module`        | dict        | Optional  | `None`   | Mapping from the module path regex to the layer range in `select` mode, for example, `{'.*mlp': [0-31]}`. This parameter is required when `mode=select`.                                                                                                                                                                                     |
| `exclude_op`           | dict        | Optional  | `None`   | Modules/operators to be excluded during recomputation. The format is the same as that of `select_module`. Matched modules/operators directly reuse the cached forward output during recomputation, skipping the actual computation. This parameter takes effect for both the `full` and `select` modes and is invalid when `mode=None`.|

> **Layer range format**: Write `5` for a single layer and `0-19` for a range. The layer numbers in the same list must be in ascending order and cannot overlap. The layer number cannot exceed `[0, num_layers-1]`.

**Selectively Recomputing Modules and Operators**: The keys of `select_module` are regexes of the module/operator paths within a layer, which can match both child modules and operators (callable attributes):

```yaml
# Assume that num_layers of the model is 8.
recompute:
  mode: select
  select_module:
    '.*attention': [0-3]          # Recompute the attention submodule at layers 0 to 3.
    '.*mlp': [4-7]                # Recompute the MLP submodule at layers 4 to 7.
    'self_attention.cast': [5]    # Recompute the cast operator of attention at layer 5.
recompute_comm:
  enable: False
swap:
  enable: False
```

`exclude_op` supports excluding three types of targets, which can be combined in the same dictionary:

**Excluding a specified module (cell)**: The entire child module directly reuses the cached forward output during recomputation, skipping all computation of the module.

```yaml
# Assume that num_layers of the model is 8.
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    'self_attention.linear_proj': [0-1]   # Exclude the attention output projection module at layers 0-1.
    'mlp.linear_fc2': [2-3]               # Exclude the MLP output projection module at layers 2-3.
recompute_comm:
  enable: False
swap:
  enable: False
```

**Excluding a specified ordinary operator (callable attribute)**: Function or operator attributes on the module (such as `add`, `reshape`, `cast`, `sigmoid`, and so on) retain their forward outputs during recomputation.

```yaml
# Assume that num_layers of the model is 8.
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    '.*add': [2-3]              # Globally match operator attributes named add.
    'self_attention.cast': [5]  # cast of attention at layer 5.
recompute_comm:
  enable: False
swap:
  enable: False
```

**Excluding a specified communication operator**: TP/EP communication operators retain their forward outputs during recomputation to avoid repeatedly initiating collective communication during backpropagation.

```yaml
# Assume that num_layers of the model is 8, with TP + EP enabled.
recompute:
  mode: full
  full_recompute_layer: [0-7]
  exclude_op:
    '.*allgather': [4-7]                                      # Exclude AllGather
    'self_attention.linear_proj.output.reducescatter': [4-7]  # Exclude ReduceScatter at the specified position
recompute_comm:
  enable: False
swap:
  enable: False
```

### Communication Recomputation (recompute_comm) Configuration

| Parameter            | Data Type | Required/Optional| Default Value    | Value Description                                |
|------------------|-------|------|---------|--------------------------------------|
| `enable`         | bool  | Optional  | `False` | Specifies whether to enable communication recomputation.                          |
| `select_module`  | dict  | Optional  | `None`  | Mapping from the communication operator path to the layer range. This parameter is required when `enable=True`.  |

**Recomputing the all-gather Communication Operator**

```yaml
# Assume that num_layers of the model is 8.
recompute_comm:
  enable: True
  select_module:
    '.*\.all_gather': [0-3]   # Recompute AllGather operators at layers 0 to 3.
recompute:
  mode: None
swap:
  enable: False
```

### Activation Swap (swap) Configuration

| Parameter              | Data Type | Required/Optional| Default Value    | Value Description                                                             |
|--------------------|-------|------|---------|-------------------------------------------------------------------|
| `enable`           | bool  | Optional  | `False` | Specifies whether to enable activation Swap.                                                    |
| `default_prefetch` | int   | Optional  | `1`     | Prefetch the activations of the *N*th layer ahead back to the NPU during backward computation, to hide the CPU→NPU fetch latency. |
| `layer_swap`       | list  | Optional  | `None`  | List of full-layer Swap items. Each item is `{layers: [...]}`.                              |
| `op_swap`          | list  | Optional  | `None`  | List of operator-level Swap items. Each item is `{op_name: ..., layers: [...]}`.               |

> **Swap does not support pipeline parallelism**: When `pp > 1`, enabling Swap will be directly intercepted in the configuration verification phase. If you want to save the graphics memory in the pipeline parallelism scenario, use recomputation.

**Scenario: Full-Layer Swap + Operator-Level Swap**

```yaml
# Assume that num_layers of the model is 8.
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: [0-1]        # Full-layer Swap for the first two layers
  op_swap:
    - op_name: '.*mlp'
      layers: [2-3]        # Swap for the MLP operators at layers 2 and 3
recompute:
  mode: None
recompute_comm:
  enable: False
```

### Combination Constraints

- **Intra-layer exclusion**: Full-layer recomputation and full-layer Swap cannot be configured for the same layer. Operator-level recomputation and operator-level Swap cannot be configured for the same module either. Otherwise, an error is reported during verification. During planning, ensure that the recomputation layer and the Swap layer do not overlap.
- When `recompute.mode` is not None, the corresponding `full_recompute_layer` (full) or `select_module` (select) must be provided. When `recompute_comm.enable` is True, `select_module` must be provided.

## Related Documents

- Overall training process and quick start: [Training Guide](../guide/training.md)
- Quick start example: [Quick Start](../quick_start/quick_start.md)
- Configuration file overview and field context: [Configuration File Description](./configuration.md)
- Reducing the graphics memory usage on the data side (EOD mask compression and variable-length FlashAttention): [Datasets](./dataset.md)
- Framework capability overview: [Overview](../introduction/overview.md)
