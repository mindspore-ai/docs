# Distributed Parallel Training

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/parallel_training.md)

When the model size exceeds the capacity of a single device, dynamic graph (PyNative) training uses **multi-dimensional hybrid parallelism** to split the model and data into multiple devices for collaborative training in different dimensions. Multiple splitting methods can be combined as required: data splitting (data parallelism/FSDP), intra-operator weight splitting (tensor parallelism, TP), sequence splitting (context parallelism, CP), model layer splitting (pipeline parallelism, PP), and MoE expert splitting (expert parallelism, EP). All parallel configurations are written in the `parallelism` field of the configuration file.

These parallel capabilities are provided by [HyperParallel](https://atomgit.com/mindspore/hyper-parallel/) and are required dependencies for dynamic graph training. The version of MindSpore must be 2.10 or later. For details about the installation method, see [Installation Guide > Installing HyperParallel](../installation.md#installing-hyperparallel-required-for-dynamic-graph-training).

The usage sequence of this page is as follows:

1. **Selecting dimensions**: Select the parallel dimensions to be enabled based on the model scale and graphics memory by referring to "Parallel Capabilities."
2. **Setting parameters**: Set the degree of parallelism and batch size by referring to "Parallelism Configuration Methods."
3. **Verifying the startup**: Complete the YAML configuration by referring to the scenario examples of each dimension. Start training after confirming that parallel configuration limitations are met.

## Parallel Capabilities

The following table lists the parallel dimensions supported by dynamic graphs. You can select the dimensions to be enabled based on the model scale and graphics memory.

| Dimension         | Field                           | Sharding Object                              | Typical Scenario                         |
|-------------|---------------------------------|------------------------------------|-------------------------------|
| Data parallelism/FSDP| `data_parallel_shard`           | Sample dimension + parameter/gradient/optimizer state sharding.               | **Foundation for almost all training**. Increase the number of shards when the graphics memory is insufficient.     |
| HSDP (replica + shard)| `data_parallel_shard` (smaller than the data parallelism dimension)| Intra-node sharding and cross-node replication.                       | Multi-node training, requiring intra-node bandwidth for sharding and inter-node replication.      |
| TP    | `tensor_parallel`               | Intra-operator weight (attention/FFN linear layer).                | LLMs whose single-layer weights cannot be accommodated on a single device.              |
| CP   | `context_parallel`              | Sequence dimension (including attention computing).                       | **Long sequence** (long context/long document), when the activated graphics memory exceeds the capacity of a single device.|
| PP   | `pipeline_parallel`             | Model is divided into multiple stages by layer.                    | The model has a large number of layers and cannot be accommodated on a single device. It is often used in combination with TP/DP.  |
| EP    | `expert_parallel`               | MoE experts are distributed to different devices.                      | MoE models (such as DeepSeek-V3).        |
| SP    | Automatically enabled with TP.                    | Sequence dimension of the TP-unsharded part (LayerNorm/Dropout/residual).| Automatically takes effect when TP is enabled. It cannot be configured separately.        |

## Parallelism Configuration Methods

Distributed parallelism configuration involves two types of parameters: **degree of parallelism** and **batch size**, which are written in different fields of the configuration file.

**1. Degree of parallelism (written in the `parallelism` field)**

Select the dimensions to be enabled based on the table under "Parallelism Capabilities" and set the corresponding degree of parallelism. For dimensions that are not enabled, retain the default value `1`.

| Configuration Item                  | Description                                                                      |
|-----------------------|--------------------------------------------------------------------------|
| `data_parallel_shard` | Data parallelism sharding degree. The default value is `-1`, indicating that all remaining devices are used for sharding (pure FSDP). If this parameter is set to a positive number less than the data parallelism dimension, HSDP (intra-node sharding + cross-node replication) is formed.|
| `tensor_parallel`     | TP.                                                                   |
| `context_parallel`    | CP.                                                                  |
| `pipeline_parallel`   | PP.                                                                  |
| `expert_parallel`     | EP (required only for MoE models).                                                       |

**2. Batch size (written in the `training` field)**

| Configuration Item                | Description                      |
|---------------------|--------------------------|
| `global_batch_size` | Total number of samples consumed in a training step (one optimizer update). |
| `local_batch_size`  | Number of samples processed in a forward pass on each device.           |

After the configuration is complete, the framework automatically infers values such as the data parallel dimension and the number of gradient accumulation steps, and checks whether the total number of devices matches the batch size. For details about the inference and limitation rules, see [Parallelism Configuration Limitations](#parallelism-configuration-limitations).

## Global Configuration Example and Field Summary

The following is a configuration overview that covers the main `parallelism` fields for reference. In practice, you can tailor the configuration based on the scenario examples provided in the following.

```yaml
parallelism:
  # -- Data parallelism/FSDP/HSDP --
  data_parallel_shard: -1                 # The only DP field that needs to be configured by the user. -1 indicates that all remaining devices are used for sharding (pure FSDP).
  reshard_after_forward_policy: "default" # Specify whether to reshard devices after forward pass. The options are always, never, and default.
  cpu_offload: False                      # Offload the sharded parameters/gradients to the CPU.
  disable_gradient_division: True         # Disable automatic gradient division of FSDP (aggregation by sum).

  # -- TP --
  tensor_parallel: 1

  # -- CP --
  context_parallel: 1
  context_parallel_method: "colossal"     # colossal / ulysses / hybrid
  context_parallel_async: False
  ulysses_degree_in_cp: null              # Set the value based on the rule when Ulysses or hybrid mode is used.
  context_parallel_mask_type: "causal"

  # -- PP --
  pipeline_parallel: 1
  pipeline_parallel_interleave_num: 1     # Number of interleaved model blocks in each stage.
  pipeline_parallel_layers_per_stage: null   # Required when PP is greater than 1. The value can be auto (even distribution) or list of layer ranges for each stage. For details, see section 4.4.
  pipeline_parallel_overlap_p2p: False
  pipeline_parallel_overlap_b_f: False

  # -- EP (MoE) --
  expert_parallel: 1                      # The value must be exactly divisible by dp_shard * CP * TP. (The expert shard grid efsdp is computed based on this value.)
  moe_token_dispatcher_type: "alltoall"   # alltoall / alltoall_deredundancy
  npu_nums_per_device: 8

  # -- SP --
  # SP is automatically enabled with TP. Currently, it cannot be disabled separately and does not need to be configured here. (For details, see section 6.)
```

## 1 DP and FSDP/HSDP

### 1.1 Overview

By default, dynamic graphs use the **full-shard data parallelism (FSDP)** solution. In a shard group, parameters, gradients, and optimizer states are sharded. During forward and backward propagation, all-gather weights and reduce-scatter gradients are performed as required. The framework further divides data parallelism into two layers:

- **`dp_shard` (shard degree)**: Parameters, gradients, and optimizer states are sharded within a group, saving graphics memory. All-gather and reduce-scatter communications are performed within the group.
- **`dp_replicate` (replica degree)**: Replicate the entire weight between groups and synchronize gradients only in the backward pass. This is classic data parallelism, which involves less communication but does not save graphics memory.

If both are greater than 1, it is **hybrid sharded data parallelism (HSDP)**: intra-node sharding and cross-node replication, saving graphics memory and reducing cross-node communication. FSDP is enabled as long as `dp_shard > 1` or `cp > 1` is met.

### 1.2 Application Scenarios and Selection

- **Baseline**: For single-node multi-device training, use pure FSDP (`data_parallel_shard: -1`, with a replica degree of 1) to shard all data across devices, achieving the highest graphics memory utilization.
- **Insufficient graphics memory**: Increase the sharding degree first (that is, add more devices to the same shard group). If the graphics memory is extremely insufficient, use `cpu_offload` and `reshard_after_forward_policy: always`.
- **Reducing cross-node communication for multiple nodes**: Use HSDP and set the sharding degree to the number of devices on a node. In this case, the framework can infer that the replica degree is equal to the number of nodes. Sharded communication is completed through high bandwidth within a node, and only gradient all-reduce is performed in the replica. You only need to set `data_parallel_shard` to a positive number smaller than the data parallelism dimension.

### 1.3 Fields

| Parameter                          | Data Type | Required/Optional| Default Value        | Description                                                                                                                                          |
|--------------------------------|-------|------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_parallel_shard`          | int   | Optional  | `-1`        | Shard degree (dp_shard). `-1` indicates that all remaining devices are used for sharding (pure FSDP, with a replica degree of 1). If a positive integer is entered, the framework infers the replica degree based on this value, forming HSDP.                                                                       |
| `reshard_after_forward_policy` | String  | Optional  | `"default"` | Specifies whether to reshard devices after the forward pass. The options are `always`, `never`, and `default`. If `never` is used, the communication is performed by using the graphics memory (the aggregation weight is retained after the forward pass, and the all-gather operation is performed once less in the backward pass). If `always` is used, the aggregation weight is resharded. `default` is equivalent to `always` when PP is not used and `never` when PP is used.|
| `cpu_offload`                  | Boolean | Optional  | `False`     | Specifies whether to offload the sharded parameters/gradients to the CPU memory to further save the graphics memory. The cost is H2D/D2H copy.                                                                                                  |
| `disable_gradient_division`    | Boolean | Optional  | `True`      | Specifies whether to disable automatic gradient averaging of FSDP: aggregates gradients using **sum** instead of mean (setting **reduce op** to **"sum"** for all HSDP submodules).                                                                       |

**Related issues**:

- `disable_gradient_division` is set to `True` by default, indicating that FSDP gradients are aggregated using **sum** instead of averaging. The framework automatically corrects the order-of-magnitude difference in loss scaling (coefficient `1/(world_size//pp)`) during backpropagation and gradient norm calculation, eliminating the need for users to manually set any scaling coefficient.
- `reshard_after_forward_policy: default` **is parsed as `never` in PP**. This is because each microbatch in PP repeats the forward pass. If resharding is performed for each microbatch, repeated all-gather operations will occur. Therefore, the aggregation weight is retained by default.

### 1.4 Scenario Configuration

#### Scenario A: Single-Node 8-Device Pure FSDP (Recommended Baseline)

All 8 devices are sharded, and the replica degree is 1. `dp_replicate(1) * dp_shard(8) * cp(1) * tp(1) * pp(1) == 8` is met.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 8.
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
  expert_parallel: 1
```

#### Scenario B: 2-Node 16-Device HSDP (Intra-Node Sharding + Cross-Node Replication)

Each node has 8 devices, and intra-node sharding and cross-node replication are required. Set the sharding degree to 8. The framework infers the results to `data_parallel = 16`, `dp_replicate = 16 // 8 = 2`, and `dp_shard = 8`. `2 * 8 * 1 * 1 * 1 == 16` is met.

```yaml
parallelism:
  data_parallel_shard: 8           # 8-device sharding within a node; the replica degree 2 is inferred by the framework.
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
```

## 2 TP

### 2.1 Overview

`tensor_parallel` specifies the TP scale and performs column/row splitting on the linear layer weights in the attention and FFN **within the operator** (using `ColwiseParallel`/`RowwiseParallel` for `apply_non_moe_tp`). Within the group, activations are synchronized using all-gather/all-reduce.

### 2.2 Application Scenarios and Selection

- This mode is enabled when the single-layer weights cannot be accommodated on a single device or the attention/FFN activation of a single device is too large.
- TP communication occurs **in the forward and backward passes of each layer** and is extremely sensitive to bandwidth. **It is usually enabled only within a single node** (the TP group size is generally less than or equal to the number of devices on the node, such as 2, 4, or 8).
- Orthogonal to FSDP: `world_size = dp_total * tp * pp * cp`. After TP is enabled, the remaining devices are automatically used for data parallelism.

### 2.3 Fields

| Parameter              | Data Type| Required/Optional| Default Value | Description             |
|--------------------|------|------|------|-------------------|
| `tensor_parallel`  | int  | Optional  | `1`  | TP scale. `1` indicates that the function is disabled. |

### 2.4 Scenario Configuration: Single-Node 8-Device DP + TP Fine-Tuning

`tp=2`. The remaining `data_parallel = 8 // 2 = 4` is used for sharding (`data_parallel_shard: -1` is inferred to `dp_shard = 4`). `1 * 4 * 1 * 2 * 1 == 8` is met.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 4.
  tensor_parallel: 2
  context_parallel: 1
  pipeline_parallel: 1
  expert_parallel: 1
  reshard_after_forward_policy: "default"
  disable_gradient_division: True
```

## 3 CP

### 3.1 Overview

`context_parallel` shards the input along the **sequence dimension** so that each device processes only a segment of the sequence, significantly reducing the activation memory usage and attention computation workload for long sequences. FlashAttention exchanges keys and values (KVs) within the CP group to complete full-sequence attention. Dynamic graphs support three implementation methods (`context_parallel_method`).

| Method        | Principle                        | Limitation (`_validate_cp_method`)                                                                                                                              |
|------------|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `colossal` | (Default) Sequence-based sharding and intra-group KV exchange.          | No additional limitations. `ulysses_degree_in_cp` is processed as 1 internally.                                                                                                                  |
| `ulysses`  | All-to-all sharding in the attention head dimension.| `ulysses_degree_in_cp == context_parallel` is required. If it is not set, `context_parallel` is used by default. `num_attention_heads % ulysses_degree == 0` is also required for `context_parallel_async=True`.|
| `hybrid`   | Colossal and Ulysses hybrid     | `ulysses_degree_in_cp` must be set and meet the following conditions: `1 < ulysses_degree_in_cp < context_parallel` and `context_parallel % ulysses_degree_in_cp == 0`                          |

### 3.2 Application Scenarios and Selection

- It is mainly used for **long sequences/long contexts** (large `seq_length` and activated graphics memory exceeding the capacity of a single device). A larger CP group size results in shorter sequences per device and less activated graphics memory, but also more intra-group KV exchange communication.
- `colossal`: preferred for general use, with direct sequence sharding and no limitation on head division.
- `ulysses`: performs sharding in the head dimension, making it more suitable for models with a large number of heads. When async is enabled, ensure that the number of heads is exactly divided by the Ulysses degree.
- `hybrid`: shards the CP into two layers: sequence block and head sharding. This is suitable for ultra-long sequence scenarios where both long sequence sharding and head dimension parallelism are required.

**CP and FSDP coupling**: When CP is enabled, **FSDP also applies to the CP group**. The shard grid name is `fsdp` and the scale is `dp_shard * cp`, even if `dp_shard == 1` (see the `fsdp_enabled`/`fsdp` attribute in [parallel_dims.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/parallel_dims.py)). That is, enabling CP automatically shards parameters within the CP group, and no separate configuration is required.

In addition, before the forward pass starts, CP shards the input of a batch (such as `input_ids`, `position_ids`, and mask) along the sequence dimension and distributes the shards to each device in the CP group. In this way, each device obtains only the sequence segment it is responsible for. This step is called **CP input preparation**. Currently, it **supports only** `context_parallel_mask_type: causal`. If other values are passed, `NotImplementedError` is thrown. User-defined `attention_mask` is not accepted. CP depends on the **compressed attention mask** on the model side, and mask compression must be enabled as described in section 3.3.

### 3.3 Attention Mask Compression (Required for CP)

**Attention mask compression must be enabled when `context_parallel > 1` is met**. Otherwise, `apply_context_parallel_model_io` in [context_parallel.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/context_parallel.py) will directly throw an error:

```text
Context parallel (context_parallel > 1) requires a compressed attention mask.
Please enable use_attn_mask_compression for non-eod data, or
create_compressed_eod_mask for eod data.
```

**Reason for compression (graphics memory overhead)**: When compression is not performed, the attention mask is a `seq_length × seq_length` dense Boolean matrix. CP is mainly used in **long sequence** scenarios. When `seq_length` is large, the dense mask itself occupies a large amount of graphics memory (contradicting the goal of CP to reduce the activated graphics memory), and each CP rank needs to hold and shard the mask. The compressed mask retains only the compact information (such as the length of each subsequence) required to generate the causal/EOD mask, which is instantaneously reconstructed by the model within the attention operator, **avoiding the materialization of dense matrices**. Therefore, CP requires compression and recommends disabling dense mask construction on the dataset side.

There are two configuration methods based on the **dataset type**. In end-of document (EOD) datasets, to improve throughput, multiple short documents are **packed** into the same fixed-length sequence during pre-training, and documents are separated by a **special token EOD**. Additionally, `reset_position_ids`/`reset_attention_mask` is used to ensure that attention **does not cross document boundaries** (causal attention is performed within each document). The mask of this type of dataset is no longer a single lower triangle segment, but a segmented block diagonal structure, which needs to be compactly expressed using `create_compressed_eod_mask` to record the length of each subsequence (`actual_seq_len`). Non-EOD datasets refer to regular data that is not concatenated and where one sequence is equivalent to one document. These datasets can be compressed using a general causal mask.

- **Non-EOD datasets**: Enable `use_attn_mask_compression` (general causal mask compression) in the **`model`** field of the YAML file.

  ```yaml
  model:
    # ...
    use_attn_mask_compression: true      # Non-EOD datasets: Enable causal mask compression on the model side.
  ```

- **EOD datasets**: Enable `create_compressed_eod_mask` in the **dataset `dataloader.config`** field of the YAML file and set `create_attention_mask` to `false` (to avoid additional dense mask construction and save graphics memory).

  ```yaml
  train_dataset:
    dataloader:
      config:
        # ...
        create_compressed_eod_mask: true   # EOD datasets: Generate a compressed EOD mask (EOD mask compression is enabled on the model side).
        create_attention_mask: false       # Disable dense attention_mask construction to save graphics memory.
  ```

  `create_compressed_eod_mask: true` will be synchronized by the framework to `use_eod_attn_mask_compression` on the model side to meet the compression requirements of CP. `create_attention_mask: false` ensures that the data pipeline no longer materializes the `seq_length × seq_length` dense mask.

### 3.4 Fields

| Parameter                        | Data Type| Required/Optional| Default Value         | Description                                                                                                       |
|------------------------------|------|------|--------------|-------------------------------------------------------------------------------------------------------------|
| `context_parallel`           | int  | Optional  | `1`          | CP scale.                                                                                                   |
| `context_parallel_method`    | String | Optional  | `"colossal"` | CP implementation method. The value can be `colossal`, `ulysses`, or `hybrid`.                                                                 |
| `context_parallel_async`     | Boolean| Optional  | `False`      | Specifies whether to enable the asynchronous CP communication hook (Hyper-Parallel). When it is used together with `ulysses`, the number of heads must be exactly divided by Ulysses.                                       |
| `ulysses_degree_in_cp`       | int  | Optional  | `None`       | Ulysses dimension. When `ulysses` is used, the value must be equal to `context_parallel`. When `hybrid` is used, the value must meet `1 < Value < context_parallel` and exactly divided by `context_parallel`.|
| `context_parallel_mask_type` | String | Optional  | `"causal"`   | Mask type used for CP input preparation (by sequence dimension, see 3.2). **Only `causal` is supported.** Other values will directly throw `NotImplementedError`.                              |

### 3.5 Scenario Configuration

> The following `parallelism` examples must be used together with the mask compression configuration (in the `model` or dataset `dataloader.config` field) in section 3.3, which is omitted here.

#### Scenario A: Single-Node 8-Device Colossal CP (Long Sequence)

When `cp=2` is used, the remaining `data_parallel = 8 // 2 = 4` are used for sharding. The FSDP shard group size is `dp_shard(4) * cp(2) = 8`. `1 * 4 * 2 * 1 * 1 == 8` is met.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 4.
  tensor_parallel: 1
  context_parallel: 2
  context_parallel_method: "colossal"
  context_parallel_mask_type: "causal"
  pipeline_parallel: 1
```

#### Scenario B: Ulysses CP (Head Dimension Sharding)

`cp=4` and `ulysses` require `ulysses_degree_in_cp == context_parallel`. `1 * 2 * 4 * 1 * 1 == 8` (`data_parallel = 8//4 = 2`) is met.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 2.
  context_parallel: 4
  context_parallel_method: "ulysses"
  ulysses_degree_in_cp: 4          # It must be equal to context_parallel.
  context_parallel_async: False    # If the values is True, ensure that num_attention_heads % 4 == 0 is met.
  tensor_parallel: 1
  pipeline_parallel: 1
```

#### Scenario C: Hybrid CP (Sequence Block × Head Sharding)

`cp=4` is used and the hybrid mode requires that `1 < ulysses_degree_in_cp < cp` be met and the value must be an integer multiple of `cp`. Therefore, the value is **2**. `1 * 2 * 4 * 1 * 1 == 8` is met.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 2.
  context_parallel: 4
  context_parallel_method: "hybrid"
  ulysses_degree_in_cp: 2          # 1 < 2 < 4 and 4 % 2 == 0.
  context_parallel_mask_type: "causal"
  tensor_parallel: 1
  pipeline_parallel: 1
```

## 4 PP

### 4.1 Overview

`pipeline_parallel` shards the model **by layer into multiple stages** and works with the microbatch pipeline for execution. The dynamic graph internally uses the interleaved 1F1B scheduler (`ScheduleInterleaved1F1B`) by default. Each stage is sharded into multiple virtual model blocks through interleaving to reduce pipeline bubbles.

### 4.2 Application Scenarios and Selection

- This mode is enabled when the model has a large number of layers and cannot be accommodated on a single device. It is often used in combination with TP (intra-node) and DP (inter-node) to form 3D parallelism.
- Increasing the value of `pipeline_parallel_interleave_num` can further reduce bubbles, but it will increase the number of inter-stage P2P communications. You can also use `pipeline_parallel_overlap_p2p` to overlap communication and computation.

**Microbatch size and scheduling strategy (automatic inference/reservation)**:

- **`pipeline_parallel_microbatch_size` is automatically inferred by the framework and is not configurable by users.** [trainer.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/trainer.py) will overwrite it with the number of gradient accumulation steps: `num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)` (where `data_parallel = dp_replicate * dp_shard` is the data parallel dimension), followed by `pipeline_parallel_microbatch_size = num_accumulation_steps`. Therefore, you need to use `global_batch_size` or `local_batch_size` to indirectly control the number of microbatches (see [Configuration File Description](./configuration.md)). Manually setting this field is invalid.
- **`pipeline_parallel_schedule` (`"1f1b"` by default in `ParallelismConfig`) is not involved in the branch.** The dynamic graph always uses the interleaved 1F1B scheduling. This string is not read and is reserved. The actual pipeline behavior is determined by `pipeline_parallel_interleave_num` and `pipeline_parallel_overlap_p2p`/`pipeline_parallel_overlap_b_f`.
- **`pipeline_parallel_enable_dxdw_split` (`False` by default) is currently a reserved item.** This field is defined in `ParallelismConfig` (used for dx/dw communication sharding in PP), but is not read or used in the dynamic graph code. Modifying this field does not change the behavior. It is a reserved/placeholder item and does not need to be set.

### 4.3 Fields

| Parameter                                  | Data Type    | Required/Optional| Default Value             | Description                                                                                                    |
|----------------------------------------|----------|------|------------------|----------------------------------------------------------------------------------------------------------|
| `pipeline_parallel`                    | int      | Optional  | `1`              | Number of pipeline stages.                                                                                            |
| `pipeline_parallel_interleave_num`     | int      | Optional  | `1`              | Number of interleaved model blocks in each stage (number of virtual stages = `pp * interleave_num`). Increasing the value can reduce bubbles.                                           |
| `pipeline_parallel_layers_per_stage`   | List/String| Optional  | `None`           | Layer allocation of each stage. It is **required** when `pp > 1`. If the number of layers can be exactly divided by `pp * interleave_num`, set this parameter to `auto` (evenly placed). Otherwise, set this parameter to a list to explicitly specify the layer range of each stage (see section 4.4).|
| `pipeline_parallel_overlap_p2p`        | Boolean    | Optional  | `False`          | Specifies whether to enable overlapping of inter-stage P2P communication and computation.                                                                                 |
| `pipeline_parallel_overlap_b_f`        | Boolean    | Optional  | `False`          | Specifies whether to enable overlapping of backward (b) and forward (f) computation.                                                                                      |
| `pipeline_parallel_microbatch_size`    | int      | Optional  | `1` (automatic inference)   | The value is automatically inferred by the framework and is overwritten by `num_accumulation_steps` at runtime. You do not need to manually set it. (See the preceding note.)                                                   |
| `pipeline_parallel_schedule`           | String     | Optional  | `"1f1b"` (**reserved**)| Reserved item. Interleaved 1F1B is used for dynamic graphs. Currently, this item is not involved in branching.                                                                              |
| `pipeline_parallel_enable_dxdw_split`  | Boolean    | Optional  | `False` (**reserved**) | Reserved item. It is the switch for splitting dx/dw communication in PP. Currently, it is not read in dynamic graphs and does not affect the behavior.                                                                 |

### 4.4 Scenario Configuration

When `pp > 1` is met, you must use `pipeline_parallel_layers_per_stage` to specify the layer placement of each stage. Select either of the following methods based on whether the layers can be evenly distributed.

#### Scenario 1: Setting `auto` when layers can be evenly placed

If `num_hidden_layers` can be exactly divided by the total number of virtual stages (`pp * interleave_num`), you only need to set `auto`. The framework evenly distributes the layers in sequence across the virtual stages (each virtual stage has `num_hidden_layers / (pp * interleave_num)` layers). For example, in an 8-layer model with `pp=2` and `interleave_num=2`, each of the 4 virtual stages holds 2 layers.

```yaml
parallelism:
  data_parallel_shard: -1
  tensor_parallel: 4
  pipeline_parallel: 2
  pipeline_parallel_interleave_num: 2     # Total number of virtual stages is 4 (2 *2 = 4).
  pipeline_parallel_layers_per_stage: auto
  pipeline_parallel_overlap_p2p: True
```

If the number of layers cannot be exactly divided by the number of virtual stages and `auto` is set, an error will be reported during startup, prompting you to use explicit configuration.

#### Scenario 2: Explicitly specifying the layer range of each stage when the number of layers cannot be evenly distributed

Each item in the list corresponds to a **physical stage** (the `i`th item for `pp_rank = i`). In each item, the layer ranges are separated by commas (,). The `interleave_num`th segment is the `k`th interleaved model block of the device, corresponding to the virtual stage number `k * pp + i`. The interval is written as `"start-end"` (closed interval) or single-layer `"n"`. All intervals must exactly cover `0` to `num_hidden_layers – 1` in the virtual stage sequence, and the layer numbers must increase.

Take a 10-layer model (layer IDs 0–9) with `pp=2` and `interleave_num=2` as an example. If `10 % 4 != 0` cannot be evenly distributed, you can explicitly configure it as follows:

```yaml
parallelism:
  data_parallel_shard: -1
  tensor_parallel: 4
  pipeline_parallel: 2
  pipeline_parallel_interleave_num: 2
  pipeline_parallel_layers_per_stage:
    - "0-1, 4-8"    # Physical stage 0: Virtual stage 0 is placed on layers 0–1, and virtual stage 2 is placed on layers 4–8.
    - "2-3, 9"      # Physical stage 1: Virtual stage 1 is placed on layers 2–3, and virtual stage 3 is placed on layer 9.
  pipeline_parallel_overlap_p2p: True
```

The layers are executed in the sequence of virtual stages 0, 1, 2, and 3, that is, layers 0–1, 2–3, 4–8, and 9. The number of layers in each stage can be different (like 2/2/5/1 in this example) to balance the extra graphics memory and computation of embedding/lm_head in the first and last stages.

## 5 EP

### 5.1 Overview

For MoE models (such as DeepSeek-V3), `expert_parallel` distributes experts to different devices, and tokens are routed to the experts on the corresponding devices through all-to-all communication for expert computation (`apply_moe_ep_tp`). EP **is not an independent device dimension**. It does not appear in the product of `dp_replicate * dp_shard * cp * tp * pp == world_size`, but is **reused on the `dp_shard * cp * tp` device**. `build_mesh` of [parallel_dims.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/distributed/parallel_dims.py) constructs a separate sparse grid `["pp", "dp_replicate", "efsdp", "ep"]` for the expert, and the expert's FSDP shard grid `efsdp` is computed as follows:

```text
efsdp = dp_shard * cp * tp // expert_parallel
```

| Grid                     | Dimension                                                                    | Application Scope                 |
|-------------------------|------------------------------------------------------------------------|-----------------------|
| Dense              | `["pp", "dp_replicate", "fsdp", "tp"]`, `fsdp = dp_shard*cp`           | Non-expert parameters (attention/shared FFN, etc.)|
| Sparse (constructed during `ep > 1`)| `["pp", "dp_replicate", "efsdp", "ep"]`, `efsdp = dp_shard*cp*tp // ep`| MoE expert parameters             |

Therefore, **`expert_parallel` must be exactly divided by `dp_shard * cp * tp`**. Otherwise, `efsdp` is not a positive integer, and an error is reported when the framework constructs the grid. A larger `ep` results in a smaller `efsdp`, fewer experts per device, and less graphics memory usage, but more all-to-all communications. Whether experts are sharded with TP does not need to be configured separately. It is automatically determined by the relative value of `tensor_parallel`/`expert_parallel` in the preceding `efsdp` formula.

### 5.2 Application Scenarios and Selection

- **It is required only for MoE models.** Enable EP when there are a large number of experts and a single device cannot accommodate all experts. A larger value of `expert_parallel` results in fewer experts per device, lower graphics memory usage, and more all-to-all communications.
- **Whether experts are split with TPs is automatically determined by the relative values of `ep` and `tp`**: When `ep ≤ tp` is met, `efsdp = dp_shard*cp*tp//ep` still contains the `tp` factor, which means that the expert weight retains some TP splitting in addition to EP distribution. When `ep` increases to the point where the `tp` factor is exhausted (for example, `ep` approaches `cp * tp` or even higher), experts are primarily distributed by EP and their role in TP splitting is weakened. The limitation is always `ep` divided by `dp_shard * cp * tp`.
- **`alltoall` and `alltoall_deredundancy`**: `alltoall` uses all-to-all distribution by default. `alltoall_deredundancy` (`DeredundancyExpertParallel`) uses the redundancy elimination solution with inter-node outer all-gather/reduce-scatter + intra-node all-to-all to reduce cross-node communication. **This solution provides significant benefits only in multi-node and large-EP scenarios** and requires `expert_parallel ≥ npu_nums_per_device`. Otherwise, `ParallelismConfig.__post_init__` will report an error.

### 5.3 Fields

| Parameter                       | Data Type| Required/Optional| Default Value         | Description                                                                        |
|-----------------------------|------|------|--------------|------------------------------------------------------------------------------|
| `expert_parallel`           | int  | Optional  | `1`          | EP scale. The value must be exactly divided by `dp_shard * cp * tp` (the expert shard grid is computed based on `efsdp = dp_shard * cp * tp // ep`).|
| `moe_token_dispatcher_type` | String | Optional  | `"alltoall"` | Token distribution mode. The value can be `alltoall` or `alltoall_deredundancy`.                            |
| `npu_nums_per_device`       | int  | Optional  | `8`          | Number of NPUs on each node. This parameter is used for limitation when `alltoall_deredundancy` is set.                                     |

### 5.4 Scenario Configuration

#### Scenario A: single-node 8-device MoE with EP ≤ TP (experts reserve some TP shards)

`tp=4`, `ep=2` (`ep <= tp`). The remaining `data_parallel = 8//(4*1*1) = 2`, so `dp_shard = 2`. This satisfies the total size condition `1 * 2 * 1 * 4 * 1 == 8`; `ep(2)` divides `dp_shard*cp*tp = 2*1*4 = 8` evenly, so the expert shard grid `efsdp = 8//2 = 4`.

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 2.
  tensor_parallel: 4
  expert_parallel: 2               # It must be exactly divided by dp_shard * cp * tp (which is 8).
  moe_token_dispatcher_type: "alltoall"
  context_parallel: 1
  pipeline_parallel: 1
```

#### Scenario B: multi-node 16-device MoE with EP > TP + redundancy-free distribution

`tp=2`, `ep=8` (`ep > tp`); using deduplication-free distribution requires that `ep(8) >= npu_nums_per_device(8)`. The remaining `data_parallel = 16//(2*1*1) = 8`, so `dp_shard = 8`. This satisfies the total scale requirement `1 * 8 * 1 * 2 * 1 == 16`; `ep(8)` divides `dp_shard*cp*tp = 8*1*2 = 16` evenly, so the expert shard grid `efsdp = 16//8 = 2` (experts are primarily distributed by EP, and a portion of the TP shards is occupied by EP).

```yaml
parallelism:
  data_parallel_shard: -1          # In this case, the framework automatically infers the value to 8.
  tensor_parallel: 2
  expert_parallel: 8               # It must be exactly divided by dp_shard * cp * tp (which is 16) and is greater than or equal to npu_nums_per_device.
  moe_token_dispatcher_type: "alltoall_deredundancy"
  npu_nums_per_device: 8           # expert_parallel must be greater than or equal to this value.
  context_parallel: 1
  pipeline_parallel: 1
```

## 6 SP

Based on TP, SP further shards the parts (LayerNorm, Dropout, and residual) that are not sharded by TP by **sequence dimension** to reduce the activation memory usage of these operators.

**SP is automatically enabled with TP. Currently, SP cannot be disabled separately.** In the SPMD path of a dynamic graph, **SP is automatically applied as long as TP is enabled (`tensor_parallel > 1`)**. When the source code in [base_models/gpt/parallelize.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/base_models/gpt/parallelize.py) calls `apply_non_moe_tp`, the formal parameter `enable_sp` is always `True` and **the `sequence_parallel` field is not read**. Therefore, SP cannot be disabled through configuration.

> Note: The `sequence_parallel` field is still read by `TransformerBlock.__init__` of [transformer_block.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/transformers/transformer_block.py), but **it is used only to detect conflicts between CPs and SPs and generate alarms** (when `cp > 1` and `sequence_parallel=True`, a message is displayed indicating that the SP conflicts with the CP and the SP is ignored). It is not a function switch for SPs.

Therefore, **you do not need to configure SP separately**. After TP (`tensor_parallel > 1`) is configured by referring to section 2 on this page, SP automatically takes effect.

## Parallelism Configuration Limitations

The total scale of the device mesh must meet the following requirements. Otherwise, the framework reports an error during startup.

- **Total scale**: `dp_replicate * dp_shard * cp * tp * pp == world_size` (`world_size` indicates the total number of `msrun` devices.) `dp_replicate` (replica degree) and `dp_shard` (sharding degree) are automatically inferred by the framework. You only need to configure `data_parallel_shard`.
- **Data parallel dimension**: `data_parallel = world_size // (tp * pp * cp)`. The value must be greater than or equal to 1 and can be exactly divided by the inferred `dp_shard`.
- **Batch size**: `data_parallel * local_batch_size ≤ global_batch_size`. Otherwise, an error is reported. The framework infers the number of gradient accumulation steps based on `num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)`. Increasing the degree of parallelism reduces the value of `data_parallel`, and the number of accumulated steps increases when the batch size remains unchanged.
- **TP/SP**: They are valid only when `tensor_parallel > 1` is met. SP is automatically enabled with TP (there is no independent switch, and currently, it cannot be disabled. For details, see section 6).
- **CP method**: `ulysses` requires `ulysses_degree_in_cp == context_parallel`. `hybrid` requires `1 < ulysses_degree_in_cp < context_parallel` and `ulysses_degree_in_cp` can be exactly divided. `ulysses + async` also requires `num_attention_heads % ulysses_degree == 0`. When CP is enabled, the FSDP shard group size is `dp_shard * cp`.
- **(Required) CP mask compression**: When `context_parallel > 1` is met, mask compression must be enabled. For non-EOD data, set `use_attn_mask_compression` to `true` in the `model` field. For EOD data, set `create_compressed_eod_mask` to `true` and set `create_attention_mask` to `false` in the `dataloader.config` field of the dataset. Otherwise, the error message "requires a compressed attention mask" is displayed. For details, see section 3.3.
- **EP**: `expert_parallel` must be exactly divided by `dp_shard * cp * tp` (the expert grid `efsdp = dp_shard * cp * tp // ep` must be a positive integer). EP is not included in the product of `dp_replicate * dp_shard * cp * tp * pp == world_size`. `moe_token_dispatcher_type = alltoall_deredundancy` requires `expert_parallel ≥ npu_nums_per_device`.

## Startup

After configuring the `parallelism` and batch size and confirming that the preceding parallel configuration limitations are met, use `msrun` to start multi-device training.

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --join=True \
      run_mindformer.py --config your_config.yaml --mode 1
```

`--mode 1` indicates PyNative (dynamic graph). `--worker_num` indicates the total number of devices (`world_size`) and must be equal to the `dp_replicate * dp_shard * cp * tp * pp` inferred from the configuration. Otherwise, an error is reported during startup.

## Related Documents

- Configuration file overview, `global_batch_size`/`local_batch_size`, and gradient accumulation: [Configuration File Description](./configuration.md)
- Automatic data sharding by parallel dimension and batch size per device inference: [Datasets](./dataset.md)
- Parallel/recomputation/offload optimization related to the graphics memory: [Training Memory Optimization](./memory_optimization.md)
- Parallel scenario practices in pre-training: [Training Guide](../guide/training.md)
- End-to-end quick start: [Quick Start](../quick_start/quick_start.md)
