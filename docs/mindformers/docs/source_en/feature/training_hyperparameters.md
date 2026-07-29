# Hyperparameters and Optimizers for Training

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/feature/training_hyperparameters.md)

The hyperparameters for dynamic graph (PyNative) training are concentrated in three **top-level parallel sections** of the configuration file: `optimizer` (optimizers), `lr_scheduler` (learning rate strategies), and `training` (basic training parameters). They are parsed by `OptimizerConfig`, `LrSchedulerConfig`, and `TrainingConfig`, respectively. For details, see [Configuration File Description](./configuration.md).

This page describes the configuration and examples of optimizers, learning rate strategies, and basic training parameters, and provides combinations that can be directly used. For details about the overall training process, see [Overview](../introduction/overview.md) and [Quick Start](../quick_start/quick_start.md).

When a training is started, the configuration file is passed as `--config`. In dynamic graph mode, `--mode 1` must be explicitly specified.

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py --config xxx.yaml --mode 1" ...
```

## Quick Reference for Selection

**Optimizer selection**: Currently, dynamic graphs support `AdamW` and `Muon`.

| Dimension                | AdamW                       | Muon                                               |
|--------------------|-----------------------------|----------------------------------------------------|
| Applicable model              | General without restriction                     | **Only** models with Multi-Latent Attention enabled (such as DeepSeek-V3)|
| Whether SWAP is supported         | Supported                         | **Not supported**                                           |
| Multi-device communication              | No additional communication policy is required.                   | Two-dimensional sharding weights require all-gather or P2P aggregation (`comm_strategy`).      |
| Default `learning_rate`| `1e-5` (provided by `lr_scheduler`)| `1e-5` (provided by `lr_scheduler`)                       |
| Typical scenario              | The default scenario is preferred; general for pretrainings and fine-tunings              | DeepSeek-V3-like MLA model pretrainings, aiming for convergence quality                   |

> AdamW is the default optimizer, compatible with all models and parallel strategies. Muon is applicable only to MLA models (such as DeepSeek-V3) and cannot be enabled together with SWAP. Otherwise, an error will be reported during optimizer construction.

**Learning rate scheduler selection**: All schedulers with warmup first increase the learning rate linearly to the base learning rate. After the warmup, the behaviors vary.

| Scheduler `type`                     | Behavior After Warmup    | Typical Use Case              | Key Extension Field                                          |
|---------------------------------|-----------------|--------------------|--------------------------------------------------|
| `ConstantWarmUpLR`              | Constant           | Debugging, resumable training alignment, and short tasks       | —                                                |
| `LinearWithWarmUpLR`            | Linear decay to 0        | Simple fine-tuning              | —                                                |
| `CosineWithWarmUpLR`            | Cosine decay           | **Most commonly used for pretrainings**        | `num_cycles`, `lr_end`, and `decay_steps`             |
| `CosineWithRestartsAndWarmUpLR` | Cosine with restart         | Long training periodic restart          | `num_cycles` (number of restarts) and `decay_steps`                |
| `PolynomialWithWarmUpLR`        | Polynomial decay          | Decay curve customization          | `power`, `lr_end`, and `decay_steps`                  |
| `WarmUpStableDecayLR`           | Warmup → stable → decay (WSD)| Large-scale pretrainings, facilitating token expansion during the process| `lr_end` and `decay_start_steps`/`decay_start_ratio`|
| `CosineAnnealingLR`             | Cosine annealing (without warmup)| Simple periodic annealing            | `t_max` and `eta_min` (Note that the field name of the basic learning rate is `base_lr`.)        |

> The learning rate scheduler reuses the LR registry of `mindformers/core/lr`. In addition, variants such as `ConstantWithCoolDownLR`, `CosineAnnealingWarmRestarts`, and `LearningRateWiseLayer` are also registered. They can be enabled as required. (The registration name is subject to the `__all__` of `mindformers/core/lr/lr_schedule.py`.)

## 1. Optimizers

Both `AdamW` and `Muon` optimizers **always create fp32 master weight copies for bf16/fp16 parameters** (`_init_main_params`). The momentum/variance status and parameter update of the optimizer are completed in fp32 precision, and then written back to the low-precision model parameters, which is aligned with the mixed precision optimizer design of Megatron.

The `optimizer` section is labeled with `allow_extra = True`, which allows additional parameters (such as `use_fused`) to be passed in addition to the field table.

### 1.1 AdamW

**Application scenarios**: This is the common default optimizer, which is applicable to all models and parallelism strategies. If there is no special requirement for pretrainings and fine-tunings, `AdamW` is preferred.

```yaml
optimizer:
  type: AdamW
  betas:
    - 0.9
    - 0.95
  eps: 1.e-8
  weight_decay: 0.01
```

| Parameter                   | Data Type       | Required/Optional| Default Value          | Value Description                              |
|-------------------------|-------------|------|---------------|------------------------------------|
| `type`                  | str         | Optional  | `AdamW`       | Optimizer type.                            |
| `betas`                 | list[float] | Optional  | `[0.9, 0.95]` | Exponential decay rate of the first-order or second-order moment. The length must be 2, and each value must be within `[0, 1)`.|
| `eps`                   | float       | Optional  | `1.0e-8`      | Denominator value stability term, which must be `> 0`.                  |
| `weight_decay`          | float       | Optional  | `0.01`        | Weight decay for decoupling (AdamW-style L2 regularization), which must be `≥ 0`. |
| `weight_decay_include`  | list[str]   | Optional  | `None`        | Parameter name matching rule for forcibly applying weight decay.                 |
| `weight_decay_exclude`  | list[str]   | Optional  | `None`        | Parameter name matching rule for forcibly skipping weight decay.                |

#### Parameter Tuning Description

The recommended settings for each key parameter are as follows:

- **`betas`**: The first item controls the first-order momentum smoothing, and the second item controls the second-order moment smoothing. `[0.9, 0.95]` is commonly used for pretraining large models. For fine-tuning small data, you can set this parameter to `[0.9, 0.999]`.
- **`eps`**: This parameter is used only as a value backup and generally does not need to be modified. If a division by zero exception occurs during a training in BF16, you can increase the value of this parameter (for example, to `1e-6`).
- **`weight_decay`**: The value range for typical pretrainings is `0.01~0.1`. It works together with the `weight_decay_include` and `weight_decay_exclude` field to control the scope of effect.
- **`weight_decay_include` and `weight_decay_exclude`**: Both are parameter name matching rule lists, used to override the default decay scope. Parameters matched by `include` are forcibly included in weight decay, and those matched by `exclude` are forcibly excluded. A common practice is to put LayerNorm, bias, and embedding into `exclude` and apply weight decay only to the weights of linear layers.

  ```yaml
  optimizer:
    type: AdamW
    weight_decay: 0.1
    weight_decay_exclude:
      - "*norm*"
      - "*bias*"
  ```

### 1.2 Muon

Muon performs Newton-Schulz iterative orthogonalization on the two-dimensional or three-dimensional weights, and rolls back other weights such as word embedding and output layer to the built-in AdamW. It usually brings better convergence quality to the MLA model, but **has strict prerequisites**.

In `__init__`, Muon calls `_verify_model` and checks the SWAP configuration. If any of the following conditions is not met, `ValueError` is thrown in the **optimizer construction phase** and the training cannot be started.

1. **`model` must be passed**. Otherwise, `Model must be provided for Muon optimizer.` is reported (automatically injected by the framework when constructing the optimizer, and you do not need to manually write it in the YAML file).
2. **Legacy models are not supported**. If the model is implemented in Legacy mode, `Muon does not support Legacy Model.` is reported.
3. **Multi-latent attention must be enabled**. The value of `multi_latent_attention` must be `True`. Otherwise, `... only supports models with Multi-Latent Attention enabled.` is reported.
4. **SWAP is not supported**. `Muon does not support swap.` is reported when `swap=True` is passed.

```yaml
optimizer:
  type: Muon
  weight_decay: 0.1
  momentum: 0.95
  matched_adamw_rms: 0.2
  nesterov: True
  eps: 1.e-7
  ns_steps: 5
  ns_coefficients: [3.4445, -4.7750, 2.0315]
  adamw_betas:
    - 0.95
    - 0.95
  adamw_eps: 1.e-8
  qk_clip_enabled: True
  qk_clip_threshold: 100
  comm_strategy: allgather
```

| Parameter               | Data Type      | Required/Optional| Default Value                        | Value Description                                                                                                       |
|---------------------|------------|------|-----------------------------|-------------------------------------------------------------------------------------------------------------|
| `weight_decay`      | float      | Optional  | `0.1`                       | Weight decay.                                                                                                       |
| `momentum`          | float      | Optional  | `0.95`                      | Muon momentum coefficient.                                                                                                 |
| `matched_adamw_rms` | float      | Optional  | `0.2`                       | RMS coefficient aligned with the AdamW update magnitude.                                                                                  |
| `nesterov`          | bool       | Optional  | `True`                      | Specifies whether to use Nesterov momentum.                                                                                          |
| `eps`               | float      | Optional  | `1.0e-7`                    | Numerical stability parameter for Newton-Schulz normalization.                                                                                   |
| `ns_steps`          | int        | Optional  | `5`                         | Number of Newton-Schulz iterations (valid only for flat `ns_coefficients` format).                                                            |
| `ns_coefficients`   | tuple/list | Optional  | `(3.4445, -4.7750, 2.0315)` | Newton-Schulz coefficient, which supports flat triplets or segmented scheduling.                                                                             |
| `adamw_betas`       | tuple      | Optional  | `(0.95, 0.95)`              | `betas` used to roll back to AdamW weights.                                                                                     |
| `adamw_eps`         | float      | Optional  | `1.0e-8`                    | `eps` for rolling back AdamW.                                                                                            |
| `qk_clip_enabled`   | bool       | Optional  | `True`                      | Specifies whether to apply QK-Clip scaling to the attention logits.                                                                                |
| `qk_clip_threshold` | float      | Optional  | `100`                       | QK-Clip threshold, which must be `> 0` (verified when `qk_clip_enabled=True`).                                                            |
| `comm_strategy`     | str        | Optional  | `allgather`                 | Multi-device communication strategy. For details, see [comm_strategy Trade-off](#comm_strategy-trade-off).                                                    |
| `use_fused_adamw`   | bool       | Optional  | `False`                     | Specifies whether to use the fused AdamW operator for non-muon weights.                                                                                  |
| `adamw_include`     | list[str]  | Optional  | `None`                      | Parameters that use AdamW instead of Muon. The default value is `["*word_embeddings*", "*output_layer*"]` (Muon is used for 2D/3D weights, and AdamW is used for embeddings and output layers).|

#### Two Formats of ns_coefficients

`ns_coefficients` is normalized by `_normalize_ns_schedule` to a "one triplet per step" schedule. Two YAML formats are supported:

- **Flat triplet** `[a, b, c]` (default): The same group of coefficients is applied to all `ns_steps` iterations.
- **Segmented scheduling** `[[[a, b, c], count],...]`: Each segment repeats its own triplet `count` times. **The total number of iterations is the sum of `count` for each segment. In this case, `ns_steps` is ignored.** This format is suitable for using different coefficients for the first few steps and the last few steps (for example, the first 8 steps and the last 2 steps of DeepSeek V4).

Example of segmented scheduling:

```yaml
optimizer:
  type: Muon
  ns_coefficients:
    - [[3.4445, -4.7750, 2.0315], 8]   # Use this group of coefficients for the first 8 steps.
    - [[2.0, -1.5, 0.5], 2]            # Switch the coefficient for the last 2 steps. Total number of steps = 8 + 2 = 10.
  # Note: When segmented scheduling is used, the ns_steps field no longer takes effect.
```

#### comm_strategy Trade-off

| Value                       | Behavior                                        | Application Scenario and Cost                                                                       |
|---------------------------|--------------------------------------------|--------------------------------------------------------------------------------|
| `allgather` (default)          | Each device performs all-gather on the full weight and independently runs Newton-Schulz. | The implementation is simple. However, **redundant NS computation** occurs when there are multiple devices (the same weights are computed repeatedly on each device).                                             |
| `allgather_deredundency`  | The two-dimensional sharded weights are aggregated to a specified rank in P2P mode. NS is computed only on that rank and then distributed. | This option is enabled when **multi-device training** is performed and there are a large number of two-dimensional weights in Muon. It eliminates redundant NS computation and reduces HCCS traffic. However, it introduces P2P aggregation/distribution communication and inter-rank load allocation logic.|

> For single-device training without communication, retain the default `allgather`. For large-scale multi-device training (especially when there are many two-dimensional weights in expert/tensor parallelism), evaluate the `allgather_deredundency`.

## 2. Learning Rate Strategy (lr_scheduler)

In the `lr_scheduler` section, `type` is used to select the scheduler. `learning_rate` is the **basic learning rate after warmup**. Other fields are extended based on the scheduler type (`allow_extra = True` for this section).

> The total number of steps `total_steps` of the scheduler is automatically filled in by the framework using `training.steps` in `_build_lr_scheduler`. Users **must not** manually set `total_steps` in the `lr_scheduler` section. Otherwise, the value may be inconsistent with the actual number of training steps.
>
> `warmup_ratio` and `warmup_steps` are used to determine the number of warmup steps. `warmup_steps` indicates the number of steps. `warmup_ratio` indicates the ratio of the number of steps to the total number of steps. The framework processes data based on `_get_lr_steps`. **Set only one of them**. If `warmup_ratio` is set, `ratio × total_steps` is used. Otherwise, `warmup_steps` is used. If both of them are set, `warmup_ratio` is preferred and `warmup_steps` is ignored.

The following lists the minimum available segments of each scheduler. (The field names are subject to the `__init__` signature of each `mindformers/core/lr/lr_schedule.py`.)

### 2.1 ConstantWarmUpLR—Constant After Warmup

```yaml
lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_steps: 100        # Or use warmup_ratio.
  warmup_lr_init: 0.0
```

Special fields are only warmup-related (`warmup_steps`, `warmup_ratio`, and `warmup_lr_init`), and the value is constant after warmup.

### 2.2 LinearWithWarmUpLR—Linear Decay After Warmup

```yaml
lr_scheduler:
  type: LinearWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
```

The learning rate linearly decays to 0. The total number of steps is obtained from `total_steps` injected by the framework, and there is no additional curve field.

### 2.3 CosineWithWarmUpLR—Cosine Decay After Warmup (Most Commonly Used for Pretrainings)

```yaml
lr_scheduler:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
  lr_end: 0.0              # Cosine decay end point
  num_cycles: 0.5         # Half cycle, from the peak to lr_end
  # decay_steps: 9000     # (Optional) Customized number of decay steps. If this parameter is left blank, the value of total_steps is used.
```

The special extension fields include `num_cycles` (default value: `0.5`), `lr_end` (default value: `0.0`), and `decay_steps`/`decay_ratio`.

### 2.4 CosineWithRestartsAndWarmUpLR—Cosine with Restarts

```yaml
lr_scheduler:
  type: CosineWithRestartsAndWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  lr_end: 0.0
  num_cycles: 1.0         # Number of restarts (≥ 1)
```

Special extension fields include `num_cycles` (default value: `1.0`, indicating the number of restart periods), `lr_end`, and `decay_steps`.

### 2.5 PolynomialWithWarmUpLR—Polynomial Decay After Warmup

```yaml
lr_scheduler:
  type: PolynomialWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  lr_end: 1.e-7           # Decay end point
  power: 1.0              # Polynomial power. 1.0 indicates linear, and > 1 indicates faster decay.
```

Special extension fields include `power` (default value: `1.0`), `lr_end` (default value: `1e-7`), and `decay_steps`.

### 2.6 WarmUpStableDecayLR—Warmup/Stable/Decay (WSD)

```yaml
lr_scheduler:
  type: WarmUpStableDecayLR
  learning_rate: 1.e-5
  warmup_steps: 100
  warmup_lr_init: 0.0
  lr_end: 1.e-7
  decay_start_ratio: 0.8  # Decay starts at 80% of the total steps. Alternatively, you can specify the absolute steps using decay_start_steps.
```

Special extended fields include `lr_end` (default: `1e-7`), `decay_start_steps` (absolute start step), and `decay_start_ratio` (percentage of total steps, choose either one). The advantage of WSD is that the stable phase can be extended at any time, which allows for additional training tokens without changing the curve shape.

### 2.7 CosineAnnealingLR—Cosine Annealing (Without Warmup)

```yaml
lr_scheduler:
  type: CosineAnnealingLR
  base_lr: 1.e-5          # Note: This scheduler uses base_lr instead of learning_rate.
  t_max: 1000             # Number of steps in half of the cosine period.
  eta_min: 0.0            # Lower bound of annealing.
```

Special extended fields include `t_max` (number of steps in half of the cosine period, which must be a positive integer) and `eta_min` (minimum learning rate, which defaults to `0.0`).

> The base learning rate parameter name of this scheduler is `base_lr` (instead of `learning_rate` of other schedulers), and it does not contain the warmup phase. Pay attention to this difference during configuration migration.

## 3. Basic Training Parameters (training)

The `training` section controls global behaviors such as the number of training steps, batch size, gradient clipping, and reproducibility.

```yaml
training:
  steps: 1000           # Total number of training steps (also used as total_steps for the LR scheduler)
  local_batch_size: 2   # Batch size of a single device (per-rank)
  global_batch_size: 4  # Total number of samples processed globally per step
  max_norm: 1.0         # Gradient clipping threshold. This parameter is enabled when the value is greater than 0.
  seed: 42              # Random seed
  deterministic: False  # Switch of deterministic computing
```

| Parameter               | Data Type | Required/Optional| Default Value     | Value Description                     |
|---------------------|-------|------|----------|---------------------------|
| `steps`             | int   | Optional  | `1000`   | Total number of training steps, which is also used as the total number of steps for the learning rate scheduler.    |
| `local_batch_size`  | int   | Optional  | `1`      | Batch size per device. The value must be a positive integer.      |
| `global_batch_size` | int   | Optional  | `1`      | Total number of samples processed globally per step. The value must be a positive number.        |
| `max_norm`          | float | Optional  | `1.0`    | Global gradient clipping threshold. The value must be a positive number (typically `1.0`).|
| `seed`              | int   | Optional  | `42`     | Random seed.                    |
| `deterministic`     | bool  | Optional  | `False`  | Specifies whether to enable deterministic training.               |

### 3.1 Batch Size and Gradient Accumulation Steps

`global_batch_size`, `local_batch_size`, and `data_parallel` (data parallelism) determine **whether gradient accumulation is required in each step**.

```text
Number of gradient accumulation steps = global_batch_size // (local_batch_size × data_parallel)
```

That is, each data parallel device processes `local_batch_size` samples per step, and the optimizer is updated only when the total number of accumulated samples reaches `global_batch_size`. The framework performs integer division on `global_batch_size` by `local_batch_size × data_parallel` to derive the number of accumulated steps. (If the result is not an integer, no error is reported. Instead, the result is rounded down, and the effective global batch size decreases accordingly.) Here, `data_parallel` is the data parallelism (`= dp_replicate × dp_shard`). When pure FSDP is used (default `data_parallel_shard: -1`), it is equal to the FSDP sharding degree `data_parallel_shard`. When HSDP is enabled, `data_parallel_shard` is less than `data_parallel`. For details about the derivation and setting, see [Distributed Parallel Training](./parallel_training.md).

### 3.2 Gradient Clipping max_norm

Before each optimizer update, the global L2 norm (`_calculate_global_grad_norm`) of all parameter gradients is calculated. When the global norm exceeds `max_norm`, all gradients are scaled in place by the ratio of `max_norm/global norm` to reduce the norm to the threshold, suppressing gradient spikes and stabilizing training. For large model pretrainings, `1.0` is often used.

> **max_norm** should be kept positive. Do not use **0** or a negative value to disable clipping. The dynamic graph path does not have the clipping switch. The clipping logic is triggered when `clip_coef = max_norm/(Global norm + eps) < 1`. If `max_norm` is set to `0`, `clip_coef` is always less than 1, and the gradients are scaled to **0**, which means that the gradients are not updated. If it is set to a negative value, the gradients will be reversed. To perform weaker clipping, increase the value of `max_norm` instead of setting it to **0** or a negative value.

### 3.3 Reproducibility seed and deterministic

- `seed`: Unifies random sources such as initialization and data shuffling to ensure reproducibility for the same configuration.
- `deterministic`: After this function is enabled, deterministic operators are forcibly used to ensure that the results of multiple runs are consistent bit by bit, facilitating debugging and accuracy alignment.

> Using `deterministic` degrades performance. Deterministic computation sacrifices the parallel optimization of some operators, **significantly reducing the training throughput**. Enable this function only when checking accuracy issues or performing bit-by-bit reproduction experiments. Keep `deterministic: False` for a regular training.

## 4. Combination Examples (Complete YAML)

`optimizer`, `lr_scheduler`, and `training` are top-level parallel keys of YAML, which are at the same level as `parallelism`, `train_dataset`, and `model`. The following provides two sets of configurations that can be directly used (irrelevant sections such as datasets and models have been omitted. Please supplement them as needed).

### 4.1 AdamW General Pretraining

It is applicable to most models.

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 4
  max_norm: 1.0
  seed: 42
  deterministic: False

optimizer:
  type: AdamW
  betas:
    - 0.9
    - 0.95
  eps: 1.e-8
  weight_decay: 0.01
  weight_decay_exclude:
    - "*norm*"
    - "*bias*"

lr_scheduler:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.01
  warmup_lr_init: 0.0
  lr_end: 0.0
```

### 4.2 Muon+DeepSeek-V3 (MLA)

Muon is applicable only to models with MLA enabled. The following YAML example is used with the DeepSeek-V3 model. Note that `model.multi_latent_attention: True` is the prerequisite for Muon construction.

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 2
  max_norm: 1.0
  seed: 42
  deterministic: False

optimizer:
  type: Muon
  weight_decay: 0.1
  momentum: 0.95
  matched_adamw_rms: 0.2
  nesterov: True
  eps: 1.e-7
  ns_steps: 5
  ns_coefficients: [3.4445, -4.7750, 2.0315]
  adamw_betas:
    - 0.95
    - 0.95
  adamw_eps: 1.e-8
  qk_clip_enabled: True
  qk_clip_threshold: 100
  comm_strategy: allgather          # For multi-device training, you can use allgather_deredundency to remove redundancy.

lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 2.e-2
  warmup_ratio: 0.0
```

## 5. Related Documents

- [Configuration File Description](./configuration.md): Overview and parsing rules of each configuration section.
- [Distributed Parallel Training](./parallel_training.md): Derivation of the data parallelism, `global_batch_size`, and gradient accumulation steps.
- [Other Training Features](./other_training_features.md): Training features such as gradient accumulation and checkpoints.
- [Training Guide](../guide/training.md): How to start a training using `run_mindformer.py`.
- [Overview](../introduction/overview.md): Overall introduction to MindSpore Transformers dynamic graph.
- [Quick Start](../quick_start/quick_start.md): Complete an end-to-end training.
