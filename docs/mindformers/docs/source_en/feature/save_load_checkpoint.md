# Weight Saving and Loading

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/save_load_checkpoint.md)

For MindSpore Transformers dynamic graph (PyNative) training, weights are saved and loaded in **Safetensors** format. The framework centrally configures the saving and loading behavior in the `checkpoint` section (`CheckpointConfig`): The saving is triggered by `CheckpointCallback` of [mindformers/pynative/callback/checkpoint_callback.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/callback/checkpoint_callback.py) step by step during training, and the loading is completed by `Trainer` during the training startup phase. The underlying layers of the two modes call `mindformers.checkpoint.save_checkpoint` and `mindformers.checkpoint.load_checkpoint`, and use the `common.json` file in the weight directory to connect the resumable training.

> **Dynamic graph using only Safetensors**
>
> The dynamic graph path uses Safetensors as the only weight format. **Format conversion** between checkpoint and Safetensors is not involved, and no separate conversion script is required.

## Terms

This document involves the following terms:

- **Weight loading**

  Only model parameters are loaded, and the optimizer state and training progress are not restored. It is commonly used in scenarios such as fine-tuning, distillation, and transfer learning.

- **Resumable training**

  It has the same meaning as resumable training, but emphasizes the resumption of the training process from a specific checkpoint.

- **Checkpoint**

  It is a model state file saved during training and is used to resume training or export the model.

- **Global step**

  It is the number of optimization steps that have been executed during training, which is used for learning rate scheduling, training resumption, and training statistics.

## How to Choose Checkpoint Configurations

Select the saving and loading modes based on your actual requirements:

```text
Do you need to resume training from the interruption point?
│
├── Yes
│   └── Use the entire segment for resumable training.
│       ├── Save the optimizer state.
│       ├── Save the training progress.
│       └── Use the resume configuration during loading.
│
└── No
    │
    ├── Perform only for model inference.
    │   └── Save only the weights.
    │
    └── Perform for fine-tuning a new task.
        └── Load the model weights, but not the optimizer state.
```

Common scenario mapping

| Scenario     | Recommendation Mode |
|---------|-------|
| Recovery from training interruption | Resuming training of the entire section |
| Cluster fault recovery | Resuming training of the entire section |
| Model fine-tuning   | Weight loading |
| Model inference and deployment | Weight loading |
| Model conversion and export | Weight loading |

## Selecting Fields by Scenario

The following table helps you quickly locate the fields that need to be paid attention to in each scenario. For details about the semantics, see the corresponding sections below.

| Scenario                  | Key Field                                               | Description                                   |
|----------------------|-----------------------------------------------------|---------------------------------------|
| Disabling saving (Only the loss is displayed and the weight is not saved.)| `enable_save: False`                                | Retains only the loss/monitor. No weight is saved.             |
| Basic periodic saving              | `save_path`/`save_interleaved_steps`/`save_max` | Saves by step and retains the latest several copies.                          |
| Asynchronous saving                | `async_save: True`                                  | Overlaps between flushing and computation, reducing saving blocking.                       |
| Weight-only saving (without saving optimizer weights)      | `no_save_optim: True`                               | Occupies smaller size. However, the optimizer state cannot be saved.                      |
| Redundancy-free saving               | `remove_redundancy: True`                           | Performs multi-device segmentation and deduplication, reducing occupation.                          |
| Multi-device layout cache              | `save_global_layout_cache`                          | Reuses shard metadata to avoid recomputing each time.                       |
| Full resumable training              | `load_path` + `no_load_optim: False`                | Restores the weight, optimizer, step, and data cursor.                  |
| Fine-tuning loading only weights             | `load_path` + `no_load_optim: True`                 | Only the weight is loaded, and the optimizer starts from the beginning.                         |
| Multi-device balanced loading              | `load_balanced: True`                               | Performs shard balancing + parameter broadcast, eliminating redundant parameter loading. This is valid only in multi-device sharding scenarios.|

> **Field Ownership**
>
> The fields for saving are consumed by `CheckpointCallback`, and the fields for loading are consumed by `Trainer._load_checkpoint`. Both types of fields are written under the same `checkpoint` field and do not affect each other.

## Saving

### Triggering Mechanism

The saving logic is all in `CheckpointCallback`. The key behaviors are as follows:

- **Whether to mount the callback**: `Trainer._create_built_in_callbacks` checks `enable_save`. When `enable_save` is set to `False`, **`CheckpointCallback` is not constructed**, and only `LossCallback` and `MonitorCallback` are retained. In this case, all other saving fields are invalid.
- **Save by step**: `on_step_end` triggers the save operation once at `state.global_step % save_interleaved_steps == 0`.
- **Supplementary save upon training completion**: `on_train_end` saves **an additional copy of the final weight** when the training is complete to ensure that the final training result is not lost.
- **Deduplication**: `_last_triggered_step` is maintained internally. If the current step has the same name as the previously saved step, the previous step will be overwritten.
- **Excess clearance**: `save_max` specifies the maximum number of copies that can be retained. If the number of copies exceeds the limit, the earliest directory is deleted by time. However, only the weights saved in the current training round are deleted.
- **Path verification**: If `save_path` is empty, `ValueError("save_path must be provided for CheckpointCallback.")` is thrown directly during callback construction. Therefore, `save_path` must be configured when saving is enabled.

**Application scenarios**: The saving function is used to persist model weights and optimizer states during training. Enable saving when resumable training, model deployment, or fine-tuning downstream tasks is required. If you only need to view training metrics (loss) and do not need to save weights, you can disable saving (`enable_save=False`) to skip the overhead of flushing weights to disk.

> **Output Directory Structure**
>
> Each time data is saved, a subdirectory named by step is generated in `save_path`, containing:
>
> - Safetensors weight shards (model shards; `no_save_optim` specifies whether optimizer shards are included).
> - `common.json`: metadata of the fine-tuning (see [Loading](#loading)).
> - Shard layout metadata `metadata.json`.

### Fields

| Parameter                      | Data Type | Required/Optional| Default Value           | Description                                                                             |
|----------------------------|-------|------|----------------|-----------------------------------------------------------------------------------|
| `enable_save`              | Boolean | Optional  | `True`         | Specifies whether to enable weight saving. If this parameter is set to `False`, the weight is not saved.                                                       |
| `save_path`                | String  | Optional  | `""`           | Specifies the save directory. This parameter is required when saving is enabled. If this parameter is left empty, `ValueError` is reported.                                                   |
| `save_max`                 | int   | Optional  | `5`            | Specifies the maximum number of weights to be saved. If the number of weights exceeds the maximum, the earliest weights are deleted by time. Only the weights saved for the current training are deleted each time. (Special scenario: If a checkpoint has been saved for the current step, the directory corresponding to the step is overwritten.)|
| `save_interleaved_steps`   | int   | Optional  | `1000`         | Specifies the number of steps between each save. The save is triggered when the number of steps is an integer multiple of `global_step`.                                                   |
| `no_save_optim`            | Boolean | Optional  | `False`        | If this parameter is set to `True`, only the model weights are saved, and the optimizer state is not saved.                                                      |
| `async_save`               | Boolean | Optional  | `False`        | If this parameter is set to `True`, asynchronous saving is enabled, and flushing to disk overlaps with computation.                                                        |
| `prefix`                   | String  | Optional  | `"checkpoint"` | Specifies the prefix of the saved file name.                                                                         |
| `remove_redundancy`        | Boolean | Optional  | `False`        | If this parameter is set to `True`, redundant data between multi-device shards is removed.                                                          |
| `save_global_layout_cache` | Boolean | Optional  | `True`         | If this parameter is set to `True`, the global shard layout of multiple devices is cached to avoid recomputing the shard metadata each time the data is saved.                                              |

### Scenario-Based Configuration

#### Scenario 1: Basic Periodic Saving

Most commonly used configuration: Save a replica of data at a fixed step interval and retain the latest several replicas. This configuration is suitable for most training tasks.

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000   # Save every 1000 steps.
  save_max: 5                    # Retain a maximum of 5 replicas. If the number of replicas exceeds 5, the earliest replicas are deleted.
  prefix: "checkpoint"           # Prefix of the saved weight name.
  no_save_optim: False           # Save the optimizer state to facilitate resumable training (restoring the weight, optimizer state, and training progress).
  async_save: False
  remove_redundancy: False
  save_global_layout_cache: True
```

> Application scenarios: single-data-source and single-/multi-device common training. Cost: Each save operation blocks training until the data is flushed to the disk. If the number of steps is large and the save operation is frequent, you can use [asynchronous saving](#scenario-2-asynchronous-saving-reducing-saving-blocking).

#### Scenario 2: Asynchronous Saving (Reducing Saving Blocking)

When the weight is large and synchronous flushing significantly slows down the training, enable `async_save`. Before saving, the framework calls `AsyncSaveManager.prepare_before_save` to overlap the disk write operation with subsequent training computation.

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  async_save: True               # Enable asynchronous saving.
```

> Applicable scenarios: LLMs, frequent saving, and slow disks. Cost: Saving is performed in the background, which consumes additional memory/thread resources. If the process exits unexpectedly, the latest asynchronous saving operation may not be completed.

#### Scenario 3: Weight-Only Saving (Without Saving Optimizer Weights)

When `no_save_optim` is set to `True`, only model weights are saved, significantly reducing the size. This is suitable for scenarios where only the weight product is required (for example, only fine-tuning is performed later).

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  no_save_optim: True            # Do not save optimizer weights.
```

> **⚠️ Impact on resumable training**
>
> If `no_save_optim` is set to `True`, the saved weights do not contain optimizer weights and cannot be used for strict restoration of optimizer momentum and second-order moment during the entire resumable training process. If you want to resume training from this weight, use `no_load_optim: True` on the loading side.

#### Scenario 4: Redundancy-Free Saving (Reducing Memory Usage)

During multi-device training, duplicate weight shards exist between different ranks. If `remove_redundancy` is set to `True`, the redundant data is removed during saving to reduce disk usage.

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  remove_redundancy: True        # Remove redundancy between shards.
```

> Application scenarios: multi-device sharding and insufficient disk space. Note that redundancy removal depends on the multi-device sharding metadata (`sharded_tensor_metas`, which is always empty for a single device). **This option does not take effect (is silently ignored) in single-device scenarios.**

#### Scenario 5: Multi-device Layout Cache

In multi-device scenarios, the sharding metadata (`sharded_tensor_metas`) of each rank needs to be collected before saving. If `save_global_layout_cache` is set to `True` (default), the metadata is cached and reused in subsequent saving operations. If this parameter is set to `False`, the cache is cleared after each saving operation and recomputed next time.

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  save_global_layout_cache: True # Cache the shard layout to accelerate subsequent saving.
```

> Application scenarios: common multi-device training where the shard layout remains unchanged during training. Remaining `True` can eliminate the overhead of recomputing each time. Set this parameter to `False` only when the shard layout may change during training and needs to be re-collected each time. In single-device scenarios, sharding is not involved, and this field is not applicable.

## Loading

### Loading Process

Loading is performed during the `Trainer.train` startup phase. The weight loading directory is obtained from `load_path`. The core process is `Trainer._load_checkpoint`.

1. **Reading metadata**: Read `CommonInfo` from `load_path/common.json` (see the following table).
2. **Restoring the step and data cursor**: When the training is resumed for the entire segment (`no_load_optim=False`), use `global_step` in `common.json` as the start point for resumable training. If the current `global_batch_size` is different from the saved one, scale `global_step = global_step * (Original global_batch_size/new global_batch_size)` by proportion, call `train_dataset.set_init_step(global_step)` to align the dataset cursor to the resumable training position, and write back the result to `state.global_step`.
3. **Weight loading (automatic resharding)**: Call `load_checkpoint`. Internally, Use`ReshardLoader` to process distributed resharding. Therefore, **the parallel layout for saving can be different from that for loading**. The framework automatically reshards to the current layout.
4. **(Optional) Balanced loading**: When `load_balanced` is set to `True`, `apply_balance_shard_strategy` is used to compute the redundant parameter mapping between ranks, and then `single_parameter_broadcast` is used to broadcast parameters between ranks. In this way, **repetitive reading of redundant parameters is eliminated**.
5. **(Optional) Optimizer master weight update**: The optimizer is not loaded when `no_load_optim` is set to `True`. After the loading is complete, `optimizer.reload_main_params_from_model()` is called to update the FP32 master weight using the newly loaded model parameters, ensuring that the master weight is aligned with the model parameters.

**Application scenarios**: The loading function is used to resume training from existing weights or initialize model parameters. This function is used when you need to resume training from the point where training was interrupted (resumable training for the entire section), fine-tune a new task based on pre-trained weights, or load weights for inference and deployment. The loading mode is specified by fields such as `no_load_optim` and `load_balanced`. You can configure the fields based on the scenario.

> If `load_path` is empty and `checkpoint_path` is not passed, no weight is loaded, and training starts from random initialization.

### Fields Recorded in `common.json`

#### Example of common.json

The typical content in `common.json` is as follows:

```json
{
  "epoch_num": 1,
  "step_num": 100,
  "global_step": 100,
  "loss_scale": "1.0",
  "global_batch_size": 2,
  "ckpt_status": null
}
```

`common.json` is written by `CommonInfo` on the saving side (`mindformers/checkpoint/checkpoint.py`) and is used to restore the step and data cursor during resumable training.

| Field                 | Description             | Function in Resumable Training                                              |
|---------------------|-----------------|------------------------------------------------------|
| `epoch_num`         | Current epoch of training.   | Metadata records.                                               |
| `step_num`          | Step number in the current epoch.  | Metadata records.                                               |
| `global_step`       | Total number of global training steps across epochs.| Start point of resumable training; `set_init_step` driven after scaling proportionally when `global_batch_size` changes.|
| `loss_scale`        | Gradient amplification coefficient.         | Metadata records.                                               |
| `global_batch_size` | Global batch size for multi-device training.     | Specifies whether to scale `global_step` compared with the current configuration.                        |
| `ckpt_status`       | Weight health status flag.       | Specifies whether the weight is healthy. The default value is `null`. When healthy weight detection is enabled, the weight health status is recorded.             |

### Fields

| Parameter            | Data Type | Required/Optional| Default Value     | Description                                            |
|------------------|-------|------|----------|--------------------------------------------------|
| `load_path`      | String  | Optional  | `""`     | Loading directory. If this parameter is left empty, no data is loaded and training starts from random initialization.                           |
| `no_load_optim`  | Boolean | Optional  | `False`  | If this parameter is set to `True`, only the model weights are loaded, the optimizer state is not loaded, and the FP32 master weights are updated.        |
| `load_balanced`  | Boolean | Optional  | `False`  | If this parameter is set to `True`, shard balancing and parameter broadcast are used to eliminate redundant parameter loading (in multi-device sharding scenarios).|

`load_balanced` applies to the following scenarios:

- The tensor parallelism (TP) scale is large.
- The pipeline parallelism (PP) scale is large.
- The data parallelism (DP) scale is large.
- There are a large number of checkpoint files.
- An obvious I/O bottleneck occurs in the loading phase.

Typical scenarios:

```text
TP=8
PP=8
DP=16
```

In large-scale distributed training, checkpoint files are usually distributed on multiple storage nodes. After `load_balanced` is enabled, the loading pressure of some ranks can be reduced, improving the overall recovery efficiency.

> **⚠️ load_worker_number does not take effect in dynamic graphs.**
>
> Although `load_worker_number` (default value: `1`) is declared in `CheckpointConfig`, **the dynamic graph loading path does not consume this field**. When `Trainer._load_checkpoint` calls `load_checkpoint`, `reshard_worker_num` is not passed, and the default value `1` is retained.
>
> Therefore, the current configuration `load_worker_number` does not change the loading parallelism. Do not rely on it to accelerate reading.

### Scenario-Based Configuration

#### Scenario 1: Full Resumable Training

The weights, optimizer status, `global_step`, and dataset cursors are restored to seamlessly continue from the interruption. This is the default form of resumable training.

```yaml
checkpoint:
  load_path: "./output/ckpt/checkpoint_5000"   # Specify the step directory where the model is saved.
  no_load_optim: False                         # Load the optimizer status.
  load_balanced: False
```

> Application scenarios: resuming training in place after an interruption. Prerequisites: `no_save_optim` has been set to `False` (including the optimizer status) when the loaded weight is saved. For details about the overall resumable training behavior, see [Resumable Training](./resume_training.md). For details about data cursor recovery, see [Datasets](./dataset.md).

#### Scenario 2: Fine-Tuning Weight-Only Loading

Only the model weight is loaded, and the optimizer is initialized from the beginning. This method is suitable for downstream fine-tuning starting from the pre-trained weight, without inheriting the optimizer momentum from the pre-training phase.

```yaml
checkpoint:
  load_path: "./pretrained/ckpt"
  no_load_optim: True              # Do not load the optimizer status.
```

> When `no_load_optim` is set to `True`, `global_step` is not scaled or resumed based on `common.json`, and the training step starts from the configured start point. After the loading is complete, the framework automatically calls `reload_main_params_from_model()` to update the FP32 master weight to avoid misalignment between the master weight and model parameters.

#### Scenario 3: Multi-device Balanced Loading

When `load_balanced` is set to `True`, the framework uses the shard balancing strategy to compute the mapping of redundant parameters between ranks, and then uses parameter broadcast to ensure that each redundant parameter is read only once. This eliminates repeated loading of redundant parameters and reduces the graphics memory and I/O required for distributed loading.

```yaml
checkpoint:
  load_path: "./output/ckpt/checkpoint_5000"
  no_load_optim: False
  load_balanced: True              # Cross-device balanced loading.
```

> **Applicable conditions**
>
> `load_balanced` uses `apply_balance_shard_strategy` for shard redistribution, **which is meaningful only in multi-device sharding scenarios**. It does not bring benefits in single-device or non-sharding scenarios.
>
> For details about the relationship between the parallel dimension and sharding, see [Distributed Parallel Training](./parallel_training.md).

## Complete Example: Resuming Training After an Interruption

### Step 1: Configuration Saving

```yaml
checkpoint:
  enable_save: True
  save_path: "./checkpoints"
  save_interleaved_steps: 1000
  no_save_optim: False
```

The following content is generated during training:

```text
checkpoints/
├── iteration_00001000/
├── iteration_00002000/
└── latest_checkpointed_iteration.txt
```

### Step 2: Training Interruption

Assume that the training exits due to a node fault at step 2300.

The latest checkpoint that is successfully saved is:

```text
iteration_00002000
```

### Step 3: Configuration Loading

```yaml
checkpoint:
  load_path: "./checkpoints"
```

### Step 4: Restoration Process

Restart training. The system automatically performs the following operations:

1. Load the model weight.
2. Load the optimizer state.
3. Restore the random number state.
4. Restore **global_step**.
5. Restore the dataset reading position.

After the restoration is complete:

```text
global_step = 2000
```

The training will continue from the position corresponding to the checkpoint, instead of starting from scratch.

### Step 5: Training Continuation

```text
2000 -> 2001 -> 2002 -> ...
```

The learning rate scheduling, optimizer momentum, and dataset cursor remain continuous.

## Related Documents

- Configuration file overview: [Configuration File Description](./configuration.md)
- Process and precautions for resumable training: [Resumable Training](./resume_training.md)
- Dataset cursor restoration involved in resumable training: [Datasets](./dataset.md)
- Background of sharding, parallelism dimension, and balanced loading: [Distributed Parallel Training](./parallel_training.md)
- Full training process: [Training Guide](../guide/training.md)
