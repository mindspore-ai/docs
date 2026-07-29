# Other Training Features

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/other_training_features.md)

In addition to hyperparameters for training such as the optimizer and learning rate, and parallelism strategies, dynamic graph (PyNative) training also has several common features: **gradient accumulation, gradient clipping, operator fusion, and hybrid precision**. This page describes the usage and configuration methods of each feature, and provides YAML sections that can be directly used.

For details about the other two types of configurations, see the corresponding documents: training hyperparameters (optimizer, learning rate, and FP32 master weight) in [Hyperparameters and Optimizers for Training](./training_hyperparameters.md) and parallel strategy in [Distributed Parallel Training](./parallel_training.md). Gradient accumulation is directly related to the parallelism degree and batch size. You are advised to read this document together with the parallelism document. For details about the complete fields, see [Configuration File Description](./configuration.md).

## Feature Check

| Feature   | Configuration Entry                      | Key Field                                                        | Default Behavior                            |
|-------|----------------------------|--------------------------------------------------------------|----------------------------------|
| Gradient accumulation | `training` / `parallelism` | `global_batch_size`, `local_batch_size`, and `data_parallel_shard`| The number of accumulated steps is inferred from the three parameters. The number of steps is not directly configured.                |
| Gradient clipping | `training`                 | `max_norm` (default: `1.0`)                                        | Enabled by default. Clipping is performed based on the global L2 norm.                |
| Operator fusion | `model`                    | `normalization`, `fused_norm`, `hidden_act`                   | `RMSNorm` + `fused_norm=True` by default.|
| Hybrid precision | `model`                    | `params_dtype`, `compute_dtype`, etc.                            | FP32 master weight + BF16 computation              |

## 1 Gradient Accumulation

Gradient accumulation is a technique that splits a global batch into several micro-batches, performs forward computation and backpropagation in sequence, and accumulates gradients until a certain number of steps are reached before updating the optimizer parameters. This allows for simulating the training effect of a large equivalent batch even when the graphics memory is limited.

### Application Scenarios

Use this method when the graphics memory is insufficient to accommodate the target global batch size at once, but you want to maintain a large equivalent batch size (to improve convergence stability).

### Inference Rules

In dynamic graphs, **the number of gradient accumulation steps is not configured separately**. Instead, it is inferred from `global_batch_size` and `local_batch_size` in the `training` segment and the data parallel dimension `data_parallel` (see `_compute_data_parallel_size` in [mindformers/pynative/trainer/trainer.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/trainer.py)):

```text
num_accumulation_steps = global_batch_size // (data_parallel * local_batch_size)
```

`data_parallel = world_size // (tensor_parallel * pipeline_parallel * context_parallel) = dp_replicate * dp_shard` indicates the complete data parallelism degree. (`dp_replicate` indicates the replica degree, corresponding to the runtime attribute `data_parallel_replicate`. `dp_shard` indicates the shard degree, that is, the parsed `data_parallel_shard`.) You only need to configure the shard degree `data_parallel_shard`. In the case of pure FSDP (with the replica degree of 1), `data_parallel` is equal to `data_parallel_shard`.

**Exact division and default inference**:

- **Round down**: The source code uses exact division `//`. If `global_batch_size` cannot be exactly divided by `data_parallel * local_batch_size`, the extra samples will be discarded. It is recommended that `global_batch_size` be exactly divided by `data_parallel * local_batch_size`.
- **The default value of `data_parallel_shard` is `-1`**. When `< 0` is used, the framework infers the value based on `dp_replicate = 1`. In this case, `data_parallel_shard` is written back as `data_parallel` (that is, the sharding degree is equal to the data parallelism dimension). You do not need to manually enter the value. This parameter is explicitly set only when HSDP (simultaneous replication and sharding) is required.

### Configuration Examples

The following example aims to accumulate 4 steps (`num_accumulation_steps = 4`). Take a single device (`data_parallel = data_parallel_shard = 1`) and `local_batch_size = 2` as an example. In this case, `global_batch_size = 8` is required: `8 // (1 * 2) = 4`.

```yaml
training:
  steps: 1000
  local_batch_size: 2        # Number of samples (one micro-batch) for each forward pass on a single device.
  global_batch_size: 8       # Number of global samples per step. The number of accumulated steps is 4 (8 // (1 * 2) = 4).
  max_norm: 1.0
  seed: 42

parallelism:
  data_parallel_shard: -1    # The default value is -1. In single-device mode, the value of data_parallel is inferred as 1.
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1

optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01
```

### Fields

| Parameter                  | Data Type| Required/Optional| Default Value  | Description                                                   |
|------------------------|------|------|-------|---------------------------------------------------------|
| `global_batch_size`    | int  | Optional  | `1`   | Number of global samples processed in each step (configuration section: `training`).                            |
| `local_batch_size`     | int  | Optional  | `1`   | Number of samples in each micro-batch on a single device (configuration section: `training`).                 |
| `data_parallel_shard`  | int  | Optional  | `-1`  | Data parallelism degree (configuration section: `parallelism`). If the value is `-1`, the value is inferred based on `data_parallel`. |

> - **Limitations**: The value of `data_parallel * local_batch_size` cannot exceed that of `global_batch_size`. Otherwise, `ValueError` is thrown, indicating that `global_batch_size` needs to be increased or `local_batch_size` needs to be decreased.
> - **Automatic loss scaling**: When `num_accumulation_steps` is greater than `1`, the loss of each micro-batch is multiplied by `1/num_accumulation_steps` and then backpropagation is performed (see `_forward_backward`) in the framework. In this way, the accumulated gradient is equivalent to the average gradient of a large batch. The scaled loss of each micro-batch is accumulated in `training_step`. Therefore, the reported `device_loss` is the average loss of the global batch (in distributed mode, the cross-device all-reduce average loss is obtained).

## 2 Gradient Clipping

Gradient clipping is a technique that sets an upper limit on the global L2 norm of all parameter gradients before the optimizer updates. When the norm exceeds the threshold, the gradients are scaled proportionally to prevent excessively large single-step updates and stabilize training.

### Application Scenarios

It is used when loss jitter or gradient norm spike occurs during training. Gradient clipping is enabled by default for dynamic graphs.

### Working Principle

Before updating the optimizer, the framework computes the global L2 norm of all parameter gradients (cross-process all-reduce synchronization in distributed mode). When the norm exceeds `max_norm`, the gradients are scaled in place proportionally (see `_calculate_global_grad_norm` in [mindformers/pynative/trainer/utils.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/trainer/utils.py)).

**Value and distributed consistency**: The norm calculation is performed at **fp32** precision to avoid numerical instability in BF16/FP16. The `Replicate` placement dimension of the DTensor is scaled and corrected to ensure that the results of multiple devices are consistent with those of a single device.

### Configuration Examples

If the loss is unstable at the early stage of LLM training, you can tighten the clipping threshold to suppress abnormal steps.

```yaml
training:
  steps: 1000
  local_batch_size: 2
  global_batch_size: 8
  seed: 42
  max_norm: 0.5      # Tighten the clipping upper limit.
```

### Fields

| Parameter       | Data Type  | Required/Optional| Default Value   | Description                                                                                                  |
|-------------|--------|------|--------|--------------------------------------------------------------------------------------------------------|
| `max_norm`  | float  | Optional  | `1.0`  | Upper limit of the global L2 norm of the gradient (configuration section: `training`). The value must be a positive number. Do not set it to `0` or a negative value. If the value is set to `0`, the gradient will be scaled to zero (equivalent to no update). If the value is set to a negative value, the gradient will be reversed. Increase the value if you need to reduce the clipping. |

## 3 Operator Fusion

Operator fusion combines adjacent operators such as normalization and activation into a single kernel for execution, reducing the kernel startup overhead in dynamic graph mode and improving the training throughput.

### Application Scenarios

By default, operator fusion is enabled for dynamic graphs and does not need to be modified. If you need to switch to the normalization type or activation function, configure it in the `model` section.

### Operators

The source code is located in [mindformers/pynative/layers/layer_norm.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/layers/layer_norm.py) and [mindformers/pynative/layers/activation.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/layers/activation.py).

| Operator       | Class               | Key Parameter (Default)                          | Behavior                                                                     |
|-----------|------------------|------------------------------------|-------------------------------------------------------------------------|
| RMSNorm   | `FusedRMSNorm`   | `eps=1e-5`, `compute_dtype=float32`| Based on fused `rms_norm` in MindSpore. It is casted to `compute_dtype` for computing, and then is casted back to the input dtype.|
| LayerNorm | `FusedLayerNorm` | <idp:inline displayname="code" id="code1198111175452">eps=1e-5</idp:inline>, <idp:inline displayname="code" id="code79811317134520">compute_dtype=float32</idp:inline>| Based on fused `layer_norm`. It is casted back to the original dtype after `compute_dtype` computing.                |
| GELU      | `GELU`           | `approximate="none"`               | The value of `approximate` can be `none` or `tanh`.                                     |
| SiLU      | `SiLU`           | `inplace=False`                    | In-place update (`inplace`) is supported.                                                      |
| SwiGLU    | `FusedSwiGlu`    | —                                  | Fused SwiGLU based on the `swiglu` operator.                                               |

**Optional values for normalization and activation:**

- **Normalization**: specified by the `normalization` field. The default value is `"RMSNorm"`. **You can set this field only to `"LayerNorm"` or `"RMSNorm"`** (parsed by `get_norm_cls`).
- **Fusion switch**: specified by the `fused_norm` field. The default value is `True`. Currently, `get_norm_cls` **supports only fusion implementation**. If `fused_norm` is set to `False`, an error will be reported. Therefore, retain the default value.
- **Activation**: specified by the `hidden_act` field. Only `gelu`, `silu`, and `fusedswiglu` are registered for `ACTIVATION_MAP`.

### Configuration Examples

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  hidden_size: 1792
  num_hidden_layers: 12
  num_attention_heads: 8
  # Operator fusion–related
  normalization: "RMSNorm"   # The options are "RMSNorm" (default) and "LayerNorm".
  fused_norm: True           # The default value is True, indicating that fusion is retained.
  rms_norm_eps: 1.e-6        # EPS of RMSNorm.
  hidden_act: "silu"         # (Optional) The value can be "gelu", "silu", or "fusedswiglu".
```

**eps field**: The default value of the `FusedRMSNorm`/`FusedLayerNorm` construction parameter `eps` is `1e-5`. The actual value (for example, `1.e-6` for DeepSeek-V3) is usually passed through model fields such as `rms_norm_eps` on the model side. The value of the `model` field is used.

## 4 Hybrid Precision

Hybrid precision training refers to using different numerical precisions for different computation layers during training. Low precision formats such as BF16/FP16 are used for main computations (such as matrix multiplication) to improve throughput, while FP32 is retained for numerically sensitive operators like normalization, softmax, and rotation position encoding (RoPE) to ensure numerical stability.

### Application Scenarios

Using BF16 for agentic computing on Ascend can significantly improve throughput. FP32 is retained for numerically sensitive operators such as normalization, softmax, and RoPE to balance speed and stability. Hybrid precision is controlled by a group of **dtype** fields in the `model` section.

**The model section is extensible**: setting `allow_extra` to `True` in the `model` section. The fields vary with the model. **The available dtype field names may vary depending on the model**. For example, `router_dense_type` is meaningful only for MoE models. These **dtype** fields do not have built-in default values in the code. The actual values come from the model YAML file. The values in the following table and example are based on the DeepSeek-V3 configuration `pynative_ds3.yaml`. Use the actual fields of the model in practice.

### Fields

| Parameter                     | Data Type| Required/Optional| Default Value         | Description                           |
|---------------------------|------|------|--------------|---------------------------------|
| `params_dtype`            | String | Optional  | `"float32"`  | Storage precision of the parameter (master weight).                   |
| `compute_dtype`           | String | Optional  | `"bfloat16"` | Agentic computing precision of matrix multiplication. Generally, BF16 is used to improve the throughput.      |
| `layernorm_compute_dtype` | String | Optional  | `"float32"`  | Normalization computation precision. FP32 is retained to ensure numerical stability.        |
| `softmax_compute_dtype`   | String | Optional  | `"float32"`  | Softmax computation precision.                  |
| `rotary_dtype`            | String | Optional  | `"float32"`  | RoPE computation precision.              |
| `router_dense_type`       | String | Optional  | `"float32"`  | MoE router computation precision (**only for MoE models**). |

**FP32 master weight replica**: When `compute_dtype` is set to **bf16** or **fp16**, the optimizer maintains an additional **FP32 master weight replica** for these parameters. The optimizer updates the weights in FP32 and then synchronizes the updates back to the low-precision parameters, preventing precision loss caused by direct accumulation of low-precision weights. For details, see [Hyperparameters and Optimizers for Training](./training_hyperparameters.md)

### Scenario-Based Configuration

**Scenario A: DeepSeek-V3 (MoE Model)**

MoE is included, and `router_dense_type` needs to be configured. All sensitive operators retain FP32:

```yaml
model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  params_dtype: "float32"            # Master weight FP32
  compute_dtype: "bfloat16"          # Agentic computing BF16
  layernorm_compute_dtype: "float32" # Normalized FP32
  softmax_compute_dtype: "float32"   # Softmax FP32
  rotary_dtype: "float32"            # RoPE FP32
  router_dense_type: "float32"       # MoE router FP32 (MoE only)
```

**Scenario B: General Dense Model**

Non-MoE models do not require `router_dense_type`. Retain the stable hybrid precision configuration for other fields.

```yaml
model:
  params_dtype: "float32"            # Master weight FP32
  compute_dtype: "bfloat16"          # Agentic computing BF16
  layernorm_compute_dtype: "float32"
  softmax_compute_dtype: "float32"
  rotary_dtype: "float32"
```

**Selection recommendation**:

- **Stability first (recommended)**: `params_dtype=float32` + `compute_dtype=bfloat16` works with FP32 master weight and FP32 normalization/Softmax/RoPE. This is the default solution with stable precision.
- **Throughput first**: More sensitive operators can be switched to BF16, but you need to pay close attention to the loss and gradient norm. Once divergence occurs, switch `layernorm_compute_dtype`/`softmax_compute_dtype` back to `float32`.

## Related Documents

- Optimizer, learning rate, and FP32 master weight: [Hyperparameters and Optimizers for Training](./training_hyperparameters.md)
- Parallel dimension (DP/TP/CP/PP) and batch size inference: [Distributed Parallel Training](./parallel_training.md)
- Dataset configuration and batch size per device: [Datasets](./dataset.md)
- Configuration file field overview: [Configuration File Description](./configuration.md)
- Full training process: [Training Guide](../guide/training.md)
- Quick start: [Quick Start](../quick_start/quick_start.md)
