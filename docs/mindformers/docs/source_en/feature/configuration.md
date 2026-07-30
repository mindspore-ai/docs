# Configuration File Description

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/feature/configuration.md)

Dynamic graph (PyNative) training uses a YAML file to manage all configurable items in a centralized manner. The **dataclass configuration system** (`TrainConfig` and its sub-configuration classes) of [mindformers/pynative/config/config.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/mindformers/pynative/config/config.py) parses and verifies the YAML file when loading it.

- Each top-level section of the YAML file corresponds to a field of `TrainConfig` and is mapped to a sub-configuration class.
- During loading, these sub-configuration classes perform type conversion and validity checks on the configuration items. If the verification fails, an error is thrown (for example, `global_batch_size` must be positive).
- When a configuration class is labeled `allow_extra = True`, extra fields that are not declared in the class can be written. The framework transparently transmits the fields to the underlying module (such as the model structure hyperparameters and scheduler hyperparameters).

## Quick Reference of Top-Level Sections and Corresponding Feature Pages

A complete configuration consists of the following 14 top-level sections (corresponding to the fields of `TrainConfig`). The following table lists the configuration class of each section, whether extra fields are allowed (`allow_extra`), and the feature page for detailed description.

| Top-Level Section             | Configuration                   | `allow_extra`              | Page for Details                                                                 |
|------------------|------------------------|----------------------------|-----------------------------------------------------------------------|
| `checkpoint`     | `CheckpointConfig`     | No                         | [Weight Saving and Loading](./save_load_checkpoint.md) and [Resumable Training](./resume_training.md)|
| `training`       | `TrainingConfig`       | No                         | This chapter                                                                  |
| `parallelism`    | `ParallelismConfig`    | No                         | [Distributed Parallel Training](./parallel_training.md)                                  |
| `optimizer`      | `OptimizerConfig`      | **Yes**                     | [Hyperparameters and Optimizers for Training](./training_hyperparameters.md)                         |
| `lr_scheduler`   | `LrSchedulerConfig`    | **Yes**                     | [Hyperparameters and Optimizers for Training](./training_hyperparameters.md)                         |
| `train_dataset`  | `TrainDatasetConfig`   | No (**Yes** for `dataloader`)| [Datasets](./dataset.md)                                                |
| `model`          | `ModelConfig`          | **Yes**                     | This chapter and README in Model Configuration                                                     |
| `monitor`        | `MonitorConfig`        | No                         | [Training Metric Monitoring](./monitor.md)                                             |
| `profiler`       | `ProfilerConfig`       | No                         | [Training Metric Monitoring](./monitor.md) (Profiling section)                               |
| `recompute`      | `RecomputeConfig`      | No                         | [Training Memory Optimization](./memory_optimization.md)                                 |
| `recompute_comm` | `RecomputeCommConfig`  | No                         | [Training Memory Optimization](./memory_optimization.md)                                 |
| `swap`           | `SwapConfig`           | No                         | [Training Memory Optimization](./memory_optimization.md)                                 |
| `callbacks`      | `List[CallbackConfig]` | The list item is **Yes**.                | This chapter                                                                  |

- `allow_extra = False`: Writing undeclared fields directly reports `Unknown configuration keys`.
- `allow_extra = True`: Undeclared fields are transparently transmitted to the underlying layer (such as the optimizer implementation, model class, scheduler, and callback registrar) as they are. This means that the fields listed in the table are **fixed fields**, and the other fields are **transparently transmitted fields extended by type**. You need to check the corresponding implementation to confirm the names and meanings of the fields.

Each sub-configuration class has a default value. Therefore, **most sections can be omitted**. If a section is omitted, the default configuration is used for the entire section (for example, if `recompute`, `swap`, and `profiler` are not specified, they are disabled by default). Only the following parameters must be explicitly provided: `model` (model type and structure), `train_dataset` (data source), and `training` (related to the training scale).

## Complete YAML Skeleton

The following is a minimum complete `train.yaml` that can be directly saved and run from `checkpoint` to `callbacks`. Comments are added to indicate which sections can be omitted and use the default values. The model section uses DeepSeek-V3 as an example. Replace it with the structure hyperparameters of the target model.

```yaml
# ===== Weight saving and loading (optional; if omitted, the weights are saved to save_path by default) =====
checkpoint:
  # Save the configuration to the default path.
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  # Load the configuration.
  load_path: ""                 # The default value is empty, indicating that no weight is loaded.
  no_load_optim: False          # For resumable training, this parameter must be set to False.

# ===== Training scale (Explicit provision is recommended) =====
training:
  steps: 1000
  local_batch_size: 1
  global_batch_size: 8
  max_norm: 1.0
  seed: 42

# ===== Multi-dimensional parallelism (optional; if omitted, only the pure FSDP strategy is used) =====
parallelism:
  data_parallel_shard: -1                      # -1 indicates automatic sharding based on the number of available devices.
  tensor_parallel: 1
  pipeline_parallel: 1
  context_parallel: 1
  expert_parallel: 1
  # sequence_parallel: False # Automatically enabled with TP. Currently, it cannot be disabled. (You do not need to configure this item. For details, see section 6 in the document "Parallel Training".)

# ===== Optimizer (The type is fixed, and other parameters are transparently transmitted based on the optimizer type.) =====
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01

# ===== Learning rate scheduling (type/learning_rate is fixed, and other parameters such as warmup is transparently transmitted based on the scheduler type) =====
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0                              # Transparently transmitted field. For details, see the implementation of the specific learning rate scheduler.

# ===== Datasets (The data source must be provided) =====
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: False
    config:                    # For details about the complete fields, see the document "Datasets".
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

# ===== Model (required): model_type/architectures is fixed, and other structure hyperparameters are transparently transmitted. =====
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

# ===== The following sections can be omitted. If omitted, the corresponding function is disabled. =====
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

Save the preceding content as `train.yaml` and run the commands in section [Loading and Startup](#loading-and-startup) to start the training.

## checkpoint—Weight Saving and Loading

Controls the saving and loading of the weight and optimizer status, including the save path, save frequency, number of retained weights, whether to perform asynchronous saving, and loading behavior during resumable training.

Weight flushing must be configured for all training tasks. In scenarios such as resumable training, incremental fine-tuning, and cross-device balanced loading, pay special attention to the adjustment of `load_path` and `no_load_optim`.

| Parameter                       | Data Type   | Required/Optional| Default Value           | Value Description                            |
|-----------------------------|---------|------|----------------|----------------------------------|
| `enable_save`               | `bool`  | Optional  | `True`         | Specifies whether to save the weight.                          |
| `save_path`                 | `str`   | Optional  | `""`           | Weight saving path.                          |
| `save_max`                  | `int`   | Optional  | `5`            | Maximum number of checkpoints that can be retained. If the number exceeds the maximum, the oldest checkpoints are deleted.    |
| `save_interleaved_steps`    | `int`   | Optional  | `1000`         | Number of steps between saving.                        |
| `no_save_optim`             | `bool`  | Optional  | `False`        | If the value is `True`, only the model weight is saved, and the optimizer status is not saved. (The training cannot be resumed accurately.)|
| `async_save`                | `bool`  | Optional  | `False`        | Asynchronous saving, reducing saving blocking.                     |
| `prefix`                    | `str`   | Optional  | `"checkpoint"` | Prefix of the saved file name.                         |
| `remove_redundancy`         | `bool`  | Optional  | `False`        | Removes distributed redundant data during saving to reduce the file size.             |
| `load_path`                 | `str`   | Optional  | `""`           | Loading path. If this parameter is left empty, no loading is performed and training starts from the beginning.                |
| `load_balanced`             | `bool`  | Optional  | `False`        | Cross-device balanced loading.                          |
| `no_load_optim`             | `bool`  | Optional  | `False`        | If the value is `True`, the optimizer status is not loaded.              |
| `load_worker_number`        | `int`   | Optional  | `1`            | Number of loading threads.                           |
| `save_global_layout_cache`  | `bool`  | Optional  | `True`         | Saves the global layout cache to accelerate subsequent loading or restoration.        |

> To implement resumable training, restore the optimizer status by ensuring that `no_load_optim: False` (default) is set. For `no_load_optim: True`, only the model weight is restored, and the training cannot be precisely resumed.

YAML configuration example:

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_max: 3
  save_interleaved_steps: 500
  async_save: True
  prefix: "qwen3"
  load_path: "./output/ckpt"     # Resumable training: points to the path where the model is saved last time.
  no_load_optim: False
```

For details about the fields, see [Weight Saving and Loading](./save_load_checkpoint.md). For details about the resumable training process, see [Resumable Training](./resume_training.md).

## training—Basic Training Parameters

Running scale parameters such as the number of training steps, batch size, gradient clipping, and random seed.

This configuration needs to be adjusted when the throughput or convergence behavior is adjusted.

| Parameter               | Data Type   | Required/Optional| Default Value     | Value Description                                                                                                                              |
|---------------------|---------|------|----------|------------------------------------------------------------------------------------------------------------------------------------|
| `steps`             | `int`   | Optional  | `1000`   | Total number of training steps.                                                                                                                             |
| `local_batch_size`  | `int`   | Optional  | `1`      | Batch size per device. The value must be a positive integer.                                                                                                                |
| `global_batch_size` | `int`   | Optional  | `1`      | Global batch size, which must be a positive number. The number of gradient accumulation steps is derived from `global_batch_size/(data_parallel × local_batch_size)`. (The value of `data_parallel` is automatically derived based on the number of used devices and other parallel configurations. You do not need to manually set this parameter.)|
| `max_norm`          | `float` | Optional  | `1.0`    | Target norm for gradient clipping. When the global gradient norm exceeds this value, the gradient is scaled proportionally.                                                                                                      |
| `seed`              | `int`   | Optional  | `42`     | Random seed.                                                                                                                              |
| `deterministic`     | `bool`  | Optional  | `False`  | Deterministic training. If this function is enabled, the results of repeated execution of the same task are the same, but the performance deteriorates.                                                                                           |

Both `global_batch_size` and `local_batch_size` must be positive numbers. Otherwise, `TrainingConfig.__post_init__` throws the `must be positive` error during loading. In addition, during training initialization, `data_parallel × local_batch_size ≤ global_batch_size` is required (an error is thrown if this condition is not met), and the number of gradient accumulation steps is derived using integer division (`global_batch_size // (data_parallel × local_batch_size)`). If the result is not an integer, no error is reported, but the value is rounded down (the effective global batch size is reduced accordingly).

> Gradient clipping is performed at each optimizer step constantly. The global gradient norm is computed first, and then `clip_coef = max_norm/(global_norm + eps)`. Under the condition of `clip_coef < 1`, the gradient is scaled proportionally. A larger `max_norm` value indicates that scaling is less likely to be triggered. However, there is no semantic meaning of disabling clipping by using a non-positive value. Therefore, keep `max_norm > 0`.

YAML configuration example:

```yaml
training:
  steps: 2000
  local_batch_size: 2
  global_batch_size: 16
  max_norm: 1.0
  seed: 1234
  deterministic: False
```

## parallelism—Multidimensional Parallelism

Declares parallelism strategies such as Fully Sharded Data Parallel (FSDP)/Hybrid Sharded Data Parallel (HSDP), tensor parallelism (TP), context parallelism (CP), pipeline parallelism (PP), expert parallelism (EP), and sequence parallelism (SP), as well as their detailed options.

In single-device or small-scale scenarios, an entire section can be omitted and the default strategy (pure FSDP) is used. In multi-device large model scenarios, there are various combinations of parallelism dimensions based on the device memory and throughput requirements.

| Parameter                                 | Data Type           | Required/Optional| Default Value                   | Value Description                                                        |
|---------------------------------------|-----------------|------|------------------------|--------------------------------------------------------------|
| `data_parallel_shard`                 | `int`           | Optional  | `-1`                   | Number of FSDP shards. `-1` indicates automatic sharding based on the number of available devices.                                   |
| `data_parallel_shard_strategy`        | `str`           | Optional  | `"optim_grads_params"` | FSDP sharding policy.                                                   |
| `reshard_after_forward_policy`        | `str`           | Optional  | `"default"`            | Specifies whether to re-shard after FSDP forward propagation. The options are `"always"`, `"never"`, and `"default"`.      |
| `cpu_offload`                         | `bool`          | Optional  | `False`                | Specifies whether to enable CPU offload, the FSDP parameter or status.                                 |
| `disable_gradient_division`           | `bool`          | Optional  | `True`                 | Specifies whether to disable automatic gradient division of FSDP (using sum instead of mean).                            |
| `tensor_parallel`                     | `int`           | Optional  | `1`                    | Tensor parallelism (TP).                                                   |
| `context_parallel`                    | `int`           | Optional  | `1`                    | Context parallelism (CP).                                                  |
| `context_parallel_method`             | `str`           | Optional  | `"colossal"`           | CP implementation method.                                                     |
| `context_parallel_async`              | `bool`          | Optional  | `False`                | Specifies whether to enable the Hyper-Parallel asynchronous CP hook.                              |
| `ulysses_degree_in_cp`                | `int` / `None`  | Optional  | `None`                 | Ulysses degree within CP. This parameter is required for hybrid CP.                                    |
| `context_parallel_mask_type`          | `str`           | Optional  | `"causal"`             | Attention mask type supported by the CP input preparation path.                                     |
| `pipeline_parallel`                   | `int`           | Optional  | `1`                    | Pipeline parallelism (number of PP phases).                                              |
| `pipeline_parallel_layers_per_stage`  | `list` / `str`  | Optional  | `None`                 | Layer allocation in each PP phase (uneven or interleaved placement supported).                                    |
| `pipeline_parallel_schedule`          | `str`           | Optional  | `"1f1b"`               | PP scheduling policy.                                                     |
| `pipeline_parallel_interleave_num`    | `int`           | Optional  | `1`                    | Number of interleaved model chunks.                                           |
| `pipeline_parallel_overlap_p2p`       | `bool`          | Optional  | `False`                | Specifies whether to enable communication overlap in the PP phase.                                        |
| `pipeline_parallel_overlap_b_f`       | `bool`          | Optional  | `False`                | Specifies whether to enable computation overlap in the PP phase.                                        |
| `pipeline_parallel_enable_dxdw_split` | `bool`          | Optional  | `False`                | Specifies whether to enable dxdw communication splitting in the PP phase.                                         |
| `sequence_parallel`                   | `bool`          | Optional  | `False`                | Sequence parallelism (SP). This field is not used in the current SPMD path. If TP is enabled, SP is automatically enabled. Currently, SP cannot be disabled independently.           |
| `expert_parallel`                     | `int`           | Optional  | `1`                    | Expert parallelism (EP, MoE).                                               |
| `npu_nums_per_device`                 | `int`           | Optional  | `8`                    | Number of NPU ranks per device.                                             |
| `moe_token_dispatcher_type`           | `str`           | Optional  | `"alltoall"`           | MoE token distribution mode. The value can be `"alltoall"` or `"alltoall_deredundancy"`.|

> Under the condition of `moe_token_dispatcher_type: alltoall_deredundancy`, `expert_parallel ≥ npu_nums_per_device` must be met. Otherwise, `ParallelismConfig.__post_init__` will throw an error. `reshard_after_forward_policy` only accepts `"always"`, `"never"`, and `"default"`. If other values are used, an error is reported during FSDP initialization.

YAML configuration example:

```yaml
# Example: FSDP + TP + PP + MoE EP
parallelism:
  data_parallel_shard: -1
  reshard_after_forward_policy: "default"
  cpu_offload: False
  disable_gradient_division: True
  tensor_parallel: 2
  pipeline_parallel: 2
  pipeline_parallel_overlap_p2p: True       # PP overlap series
  pipeline_parallel_overlap_b_f: True
  context_parallel: 1
  expert_parallel: 8                        # MoE: EP ≥ npu_nums_per_device
  npu_nums_per_device: 8
  moe_token_dispatcher_type: "alltoall_deredundancy"
```

For details about the meaning of each dimension, combination constraints, and trade-offs between the graphics memory and throughput, see [Distributed Parallel Training](./parallel_training.md).

## optimizer—Optimizers

Optimizer type and hyperparameters.

`OptimizerConfig` is marked with `allow_extra = True`. The specific implementation hyperparameters except `type` are transparently transmitted based on the optimizer type.

| Parameter                                | Data Type                   | Required/Optional| Default Value          | Value Description                                                                      |
|--------------------------------------|-------------------------|------|---------------|----------------------------------------------------------------------------|
| `type`                               | `str`                   | Optional  | `"AdamW"`     | Optimizer type (such as `"AdamW"` and `"Muon"`).                                               |
| `betas`                              | `list(float)`           | Optional  | `[0.9, 0.95]` | First-order or second-order momentum coefficient.                                                                 |
| `eps`                                | `float`                 | Optional  | `1.0e-8`      | Numerical stability parameter.                                                                     |
| `weight_decay`                       | `float`                 | Optional  | `0.01`        | Weight decay coefficient.                                                                    |
| `weight_decay_include`               | `list(string)` / `None` | Optional  | `None`        | Parameter name rule list. Weight decay is forcibly enabled for parameters that match this item.                                         |
| `weight_decay_exclude`               | `list(string)` / `None` | Optional  | `None`        | Parameter name rule list. Weight decay is forcibly disabled for parameters that match this item.                                         |

```yaml
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.1
  weight_decay_exclude: ["bias", "norm"]    # No weight decay is performed on norm/bias.
```

For details, see [Hyperparameters and Optimizers for Training](./training_hyperparameters.md).

## lr_scheduler—Learning Rate Scheduling

Learning rate scheduling policy.

`LrSchedulerConfig` declares only **two fixed fields**: `type` and `learning_rate`. Other fields (such as `warmup_ratio`, `warmup_steps`, and `min_lr`) are **transparently transmitted based on the scheduler type** and are not fixed.

| Parameter            | Data Type    | Required/Optional| Default Value    | Value Description                                                  |
|------------------|----------|------|---------|--------------------------------------------------------|
| `type`           | `str`    | Optional  | `None`  | Scheduler type (such as `"ConstantWarmUpLR"` and `"CosineWithWarmUpLR"`).|
| `learning_rate`  | `float`  | Optional  | `1e-5`  | Basic learning rate.                                                 |

In the following example, `warmup_ratio` is a non-fixed field and will be transparently transmitted to the selected scheduler `ConstantWarmUpLR`.

```yaml
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0          # Transparently transmitted field, which is parsed based on the scheduler type.
```

For details about the scheduler type list and their respective dedicated fields, see [Hyperparameters and Optimizers for Training](./training_hyperparameters.md).

## train_dataset—Datasets

Data loader and dataset configuration.

The `train_dataset` top-level field takes effect for all loaders. The `dataloader` section is labeled with `allow_extra = True`, and the fields specific to the Megatron loader (`BlendedMegatronDatasetDataLoader`) are written in this section. The data source must be provided for each training.

| Parameter                               | Data Type              | Required/Optional| Default Value                      | Value Description              |
|-------------------------------------|--------------------|------|---------------------------|--------------------|
| `dataloader`                        | `DataloaderConfig` | Optional  | `DataloaderConfig()`      | Data loader sub-configuration.          |
| `dataloader.type`                   | `str`              | Optional  | —                         | Data loader type.           |
| `dataloader.shuffle`                | `bool`             | Optional  | `False`                   | Specifies whether to shuffle data.            |
| `dataloader.column_names`           | `list(string)`     | Optional  | `["input_ids", "labels"]` | Data column name.              |
| `dataloader.python_multiprocessing` | `bool`             | Optional  | `False`                   | Specifies whether to use Python multi-process.   |
| `drop_remainder`                    | `bool`             | Optional  | `True`                    | Specifies whether to discard the tail samples that are not enough to compose a batch.|
| `num_parallel_workers`              | `int`              | Optional  | `8`                       | Number of parallel data loading workers.   |
| `prefetch_size`                     | `int`              | Optional  | `1`                       | Number of prefetch batches.             |
| `numa_enable`                       | `bool`             | Optional  | `False`                   | Specifies whether to enable NUMA affinity loading.    |

The fixed fields in `dataloader` are `type`, `shuffle`, `column_names`, and `python_multiprocessing`. Other non-fixed fields need to be transparently transmitted based on the specific loader type.

The following example configuration can be used to load a Megatron dataset:

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

For details about how to create a Megatron dataset, and the fields and scenario-specific configurations, see [Datasets](./dataset.md).

## model—Models

Model type and structure hyperparameters.

In `ModelConfig`, only `model_type` and `architectures` are fixed fields. Other structure hyperparameters (such as `hidden_size`, `num_hidden_layers`, and MoE configurations) are transparently transmitted to the model class.

| Parameter            | Data Type  | Required/Optional| Default Value    | Value Description                                |
|------------------|--------|------|---------|--------------------------------------|
| `model_type`     | `str`  | Optional  | `None`  | Model type ID (for example, `"deepseek_v3"` or `"qwen3"`). |
| `architectures`  | `str`  | Optional  | `None`  | Model architecture class name (for example, `"DeepseekV3ForCausalLM"`). |

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  # The following are structure hyperparameters that are transparently transmitted and must match the selected model.
  vocab_size: 129280
  seq_length: 4096
  hidden_size: 1792
  intermediate_size: 3072
  num_hidden_layers: 12
  num_attention_heads: 8
  use_flash_attention: True
  compute_dtype: "bfloat16"
  # MoE structure hyperparameters (if applicable)
  n_routed_experts: 8
  num_experts_per_tok: 4
```

Except the two fixed fields, all fields in the `model` section are transparently transmitted. The field names or default values are determined by the selected model class. If a field is incorrectly written, no `Unknown keys` will be reported, but the field will be ignored or an error will be reported during model construction. The structure definition of the target model will prevail.

## monitor—Training Monitoring

Training stability health check, training status monitoring, TensorBoard logs, and MoE monitoring. Each subitem has a default value, and the entire section can be omitted.

This function can be enabled when training metrics need to be visualized or flushed to disk, or training exceptions (gradient explosion, loss exception, and MoE load imbalance) need to be checked.

| Parameter                                         | Data Type                 | Required/Optional| Default Value    | Value Description                            |
|-----------------------------------------------|-----------------------|------|---------|----------------------------------|
| `train_state.local_norm`                      | `str`                 | Optional  | `""`    | Local norm name.                      |
| `train_state.local_loss`                      | `bool`                | Optional  | `False` | Specifies whether to enable local loss monitoring.                  |
| `train_state.device_norm`                     | `bool`                | Optional  | `False` | Specifies whether to enable device-level norm monitoring.                 |
| `train_state.device_loss`                     | `bool`                | Optional  | `False` | Specifies whether to enable device-level loss monitoring.                 |
| `moe_monitor.save_tokens_per_expert_interval` | `int` / `None`        | Optional  | `None`  | Interval for recording the number of tokens per expert in MoE. `None` indicates that the function is disabled.|
| `moe_monitor.target_layers`                   | `list(int)` / `None`  | Optional  | `None`  | List of target layers for MoE monitoring.                    |

YAML example:

```yaml
monitor:
  train_state:
    local_loss: True
  moe_monitor:
    save_tokens_per_expert_interval: 100
    target_layers: [0, 1, 2]
```

For details about the fields, see [Training Metric Monitoring](./monitor.md).

## profiler—Performance Analysis

Collects profile data (such as the time consumed by operators, memory, and call stacks).

`enable_profiling` is disabled by default. The entire section can be omitted. This function can be enabled when you need to locate performance bottlenecks and analyze operator distribution and graphics memory usage. In addition, you can specify a step range to avoid excessive data volume.

| Parameter              | Data Type                | Required/Optional| Default Value    | Value Description                          |
|--------------------|----------------------|------|---------|--------------------------------|
| `enable_profiling` | `bool`               | Optional  | `False` | Specifies whether to enable profiling.                |
| `start_step`       | `int`                | Optional  | `1`     | Start step, which must be a positive number.                      |
| `end_step`         | `int`                | Optional  | `1`     | End step, which must be a positive number and greater than or equal to `start_step`.   |
| `output_path`      | `str` / `None`       | Optional  | `None`  | Path for storing results, which are automatically generated when `None` is set.           |
| `profiler_rank`    | `list(int)` / `None` | Optional  | `None`  | List of ranks to be collected. `None` indicates all ranks.|
| `profiler_level`   | `int`                | Optional  | `0`     | Collection level (`0`, `1` or `2`. A larger value indicates more detailed information.)       |
| `mstx`             | `bool`               | Optional  | `False` | Specifies whether to collect lightweight dotting data.                   |
| `profile_memory`   | `bool`               | Optional  | `False` | Specifies whether to collect tensor memory data.              |
| `with_stack`       | `bool`               | Optional  | `True`  | Specifies whether to collect call stacks on the Python side.              |
| `profile_cpu`      | `bool`               | Optional  | `True`  | Specifies whether to collect CPU profiling activities.         |

The values of `start_step` and `end_step` must be positive and must meet the `start_step ≤ end_step` condition. Otherwise, `ProfilerConfig.__post_init__` throws an error.

YAML configuration example:

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

For details about the analysis method and interpretation of the generated files, see the section "Profiling" in [Training Metric Monitoring](./monitor.md).

## recompute—Recomputation

Activation checkpoint (recomputation), which uses forward recomputation to save the graphics memory. In PyNative, this is implemented by wrapping a specified layer or module with a checkpoint wrapper.

This function can be enabled when the graphics memory is insufficient. You can choose to enable full recomputation (`"full"`) or recomputation (`"select"`).

| Parameter                  | Data Type                   | Required/Optional| Default Value     | Value Description                                                                  |
|------------------------|-------------------------|------|----------|------------------------------------------------------------------------|
| `mode`                 | `str`                   | Optional  | `"None"` | Recomputation mode. The value can be `"None"`, `"full"`, or `"select"`.                             |
| `full_recompute_layer` | `list(string)` / `None` | Optional  | `None`   | This parameter is required when `mode` is set to `"full"`. It is a list of layer spec strings (such as `['0-3']` and `['0','5']`), which are in ascending order and do not overlap with each other.|
| `select_module`        | `dict` / `None`         | Optional  | `None`   | This parameter is required when `mode` is set to `"select"`. The value is in dictionary format: mapping from the module path to the list of layer spec strings.                     |

The values of `full_recompute_layer` and `select_module` are **lists of string specs**. For example, specify a single layer number as `'5'` or a range as `'0-3'` (closed interval, including endpoints, that is, layers 0, 1, 2, and 3). Enclose the value in quotation marks as `['0-3']` to prevent the bare `[0-3]` from being incorrectly parsed by YAML. The spec must be in strict ascending order and cannot overlap. The maximum layer number must be less than the number of layers in the model (valid range: `[0, num_layers-1]`).

`mode: full` example (full recomputation by layer):

```yaml
recompute:
  mode: "full"
  full_recompute_layer: ['0-3']     # Full recomputation for the first four layers
```

`mode: select` example (selective recomputation by module):

```yaml
recompute:
  mode: "select"
  select_module:
    "attention": ['0-1']            # Recompute the attention module at layers 0 and 1.
    "mlp": ['2']                    # Recompute the MLP module at layer 2.
```

For details about policy selection and memory gain, see [Training Memory Optimization](./memory_optimization.md).

## recompute_comm—Communication Recomputation

Selective recomputation is performed on communication operators to further reduce the peak value of the graphics memory. It is independent of `recompute.mode` and is controlled by its own `enable`.

This function can be enabled when the video memory needs to be compressed in addition to regular recomputation and the intermediate values stored by the communication operator occupy a large amount of memory.

| Parameter| Data Type| Required/Optional| Default Value| Value Description|
|---|---|---|---|---|
| `enable` | `bool` | Optional| `False` | Specifies whether to enable communication recomputation.|
| `select_module` | `dict` / `None` | Optional| `None` | This parameter is required when `enable` is set to `True`. Otherwise, an error is reported. The value is in dictionary format, which is the mapping from the path of the communication operator module to the list of layer spec strings.|

The following is an example:

```yaml
recompute_comm:
  enable: True
  select_module:
    "attention.qkv": ['0-3']
```

For details, see [Training Memory Optimization](./memory_optimization.md).

## swap—Activation SWAP (offload)

The activations of the transformer block or specified operator are offloaded to the CPU and then retrieved during backpropagation. This trades bandwidth for graphics memory.

This function can be enabled when the graphics memory is severely insufficient and recomputation is still insufficient. The cost is the latency of retrieving data from the CPU during backpropagation.

| Parameter              | Data Type                  | Required/Optional| Default Value    | Value Description                                                         |
|--------------------|------------------------|------|---------|---------------------------------------------------------------|
| `enable`           | `bool`                 | Optional  | `False` | Specifies whether to enable the offload of the transformer block or operator.                           |
| `default_prefetch` | `int`                  | Optional  | `1`     | Operator advance for prefetching activations before backward FlashAttention, used to mask the latency of fetching data from the CPU to the NPU.          |
| `layer_swap`       | `list(dict)` / `None`  | Optional  | `None`  | List of offload configurations for the entire layer. Each item is in the format of `{layers: ['<spec>']}`.                  |
| `op_swap`          | `list(dict)` / `None`  | Optional  | `None`  | List of operator-level offload configurations. Each item is in the format of `{op_name: <module name>, layers: ['<spec>']}`. |

In dynamic graph mode, `default_prefetch` is used to retrieve activations before executing the backward FlashAttention, masking the latency of data movement from the CPU to the NPU. Its value indicates the number of operators to be retrieved in advance. The value range is `[1, num_layers-1]` (an out-of-bounds value will trigger a verification error). `layers` in `layer_swap` or `op_swap` is also a list of quoted layer spec strings.

Example of offload for the entire layer:

```yaml
swap:
  enable: True
  default_prefetch: 1
  layer_swap:
    - layers: ['0-1']          # Offload layers 0 and 1.
```

Example of operator-level offload:

```yaml
swap:
  enable: True
  op_swap:
    - op_name: "attention"
      layers: ['0']            # Offload the attention operator at layer 0.
    - op_name: "mlp"
      layers: ['1']            # Offload the MLP operator at layer 1.
```

For details, see [Training Memory Optimization](./memory_optimization.md).

## callbacks—Training Callbacks

List of training callbacks (`List[CallbackConfig]`). Each item in the list is a callback configuration. `type` is required, and other fields are transparently transmitted based on the callback type. The built-in callbacks of the framework always take effect. This parameter is used to configure additional custom callbacks.

It is used when custom logic (such as printing, saving, and monitoring) needs to be inserted during training. If not needed, you can write `callbacks: []` or omit the entire section.

| Parameter  | Data Type | Required/Optional| Default Value   | Value Description                               |
|--------|-------|------|--------|-------------------------------------|
| `type` | `str` | Optional  | `None` | Callback type, which corresponds to the class name registered with the `CALLBACK` module. This parameter is required in the list.|
| Other fields  | —     | Optional  | —      | Transparently transmitted to constructors based on the callback type.                      |

```yaml
callbacks:
  - type: MyCustomCallback
    log_interval: 10          # Transparently transmitted to the constructor of the callback.
```

The framework calls the registrar for each item in the list to instantiate the corresponding callback. `type` must be the name of a registered callback class.

## Loading and Startup

Dynamic graph training tasks are started through [run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/run_mindformer.py). To execute a single-device task, refer to the following command:

```bash
python run_mindformer.py --config train.yaml --mode 1 --use_parallel False
```

Multi-device tasks can be executed using the [msrun_launcher.sh](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/scripts/msrun_launcher.sh) script in the **scripts** directory. The following is an example command:

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config train.yaml --mode 1" 8 # 8 indicates an 8-device task.
```

For details about the minimum running example and startup details, see [Quick Start](../quick_start/quick_start.md).

## Related Documents

- Concepts and quick start: [Model Overview](../introduction/overview.md) and [Quick Start](../quick_start/quick_start.md)
- Weight and resumable training: [Weight Saving and Loading](./save_load_checkpoint.md) and [Resumable Training](./resume_training.md)
- Data and parallelism: [Datasets](./dataset.md) and [Distributed Parallel Training](./parallel_training.md)
- Hyperparameters and graphics memory: [Hyperparameters and Optimizers for Training](./training_hyperparameters.md) and [Memory Optimization](./memory_optimization.md)
- Monitoring and analysis: [Training Metric Monitoring](./monitor.md) (including Profiling)
