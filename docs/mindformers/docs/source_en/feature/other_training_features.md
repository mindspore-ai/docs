# Other Training Features

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/other_training_features.md)

During the large-scale training of deep learning models, challenges such as memory limitations, effective utilization of computational resources, and synchronization issues in distributed training are encountered. To address these challenges, training optimization algorithms are employed to enhance training efficiency, accelerate convergence, and improve the final model performance.

MindSpore Transformers provides optimization algorithms like Recomputation, Gradient Accumulation, and Gradient Clipping for use during training.

## Gradient Accumulation

### Overview

MindSpore supported the gradient accumulation implementation interface `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` in versions after 2.1.1, which provides the gradient accumulation capability by splitting MiniBatch. MindSpore Transformers encapsulates it into a unified training process and enables it through yaml configuration. For the principle of gradient accumulation and the ability of framework measurement, please refer to [MindSpore Document: Gradient Accumulation](https://www.mindspore.cn/tutorials/en/master/parallel/distributed_gradient_accumulation.html).

### Configuration and Usage

#### YAML Parameter Configuration

To enable gradient accumulation, users only need to configure the `gradient_accumulation_steps` item under the `runner_config` item in the configuration file and set it to the required number of gradient accumulation steps:

```yaml
# runner config
runner_config:
...
gradient_accumulation_steps: 4
...
```

#### Key Parameters Introduction

| Parameter                   | Description                                                                                  | Value Description                     |
|-----------------------------|----------------------------------------------------------------------------------------------|---------------------------------------|
| gradient_accumulation_steps | The number of steps to accumulate gradients before performing backpropagation. Default: `1`. | (int, required) - Default value: `1`. |

#### Other Ways to Use Gradient Accumulation

In addition to the configuration file, when launching the `run_mindformer.py` script, you can specify the `--gradient_accumulation_steps` argument to use the gradient accumulation feature.

#### Usage Restrictions of Gradient Accumulation

> Enabling gradient accumulation will increase memory overhead. Please pay attention to memory management to prevent Out Of Memory.

1. Since the implementation of `GradAccumulationCell` relies on parallel features, gradient accumulation is currently only supported in **semi-automatic parallel mode**;
2. In addition, in the pipeline parallel scenario, the meaning of gradient accumulation is the same as micro_batch and will not take effect. Please configure the `micro_batch_num` item to increase the training batch_size.

## Gradient Clipping

### Overview

The gradient clipping algorithm can avoid the situation where the reverse gradient is too large and the optimal solution is skipped.

### Configuration and Usage

#### YAML Parameter Configuration

In MindSpore Transformers, the default training process `MFTrainOneStepCell` integrates gradient clipping logic.

You can use the following example to enable gradient clipping:

```yaml
# wrapper cell config
runner_wrapper:
type: MFTrainOneStepCell
...
use_clip_grad: True
max_grad_norm: 1.0
...
```

#### Key Parameters Introduction

| Parameter     | Description                                                                            | Value Description                         |
|---------------|----------------------------------------------------------------------------------------|-------------------------------------------|
| use_clip_grad | Controls whether gradient clipping is enabled during training, default value: `False`. | (bool, optional) - Default: `False`.      |
| max_grad_norm | Controls the maximum norm value of gradient clipping, default value: `1.0`.            | (float, optional) - Default: `1.0`. |

## GroupedMatmul

### Overview

For MoE (Mixture of Experts), there are fragmented expert computation operations and communications. The GroupedMatmul operator merges multi-expert computations to improve the training performance of MoE. By invoking the GroupedMatmul operator, multiple expert computations are fused to achieve acceleration.

### Configuration and Usage

#### YAML Parameter Configuration

To enable GroupedMatmul in MoE scenarios, users only need to configure the `use_gmm` parameter under the moe_config section in the configuration file and set it to `True`:

```yaml
moe_config:
  ...
  use_gmm: True
  ...
```

### FAQ

When using the gmm fusion operator, an error may occur if the workload is unbalanced, resulting in no tokens being assigned to an expert on a specific NPU. The error is as follows:

```log
VallueError: For primitive[Reshape]， the accumulate of x_shape must be equal to out_shape, but got x_shape: [const vector]{}, and output_shape: [const vector]{0, hiddensize}
```

In this case, you can configure `enable_gmm_safe_tokens: True` to ensure each expert is assigned at least 1 token, avoiding program errors.

```yaml
moe_config:
  ...
  enable_gmm_safe_tokens: True
  ...
```

## MoE Droprate Logging

### Overview

When training models using the MoE (Mixture of Experts) capacity scheme, certain tokens may be dropped to improve efficiency and performance. By enabling the droprate logging feature, users can monitor the occurrence rate of these drop operations in real-time during training, helping them better understand model behavior and adjust training strategies accordingly. This feature allows users to view the droprate for each layer during training. The droprate refers to the proportion of tokens dropped in a specific layer. Observing the trend of droprate changes can help users evaluate whether the current training parameters are reasonable and whether the model is effectively utilizing expert resources.

### Configuration and Usage

#### YAML Parameter Configuration

To enable the droprate logging feature, users need to configure the `callback_moe_droprate` parameter under the moe_config section in the configuration file and set it to `True`. Additionally, add the `MoEDropRateCallback` configuration item in the callback section and set model-related parameters such as `expert_num`, `capacity_factor`, `num_layers`, and `mtp_depth`. For example:

```yaml
moe_config:
  ...
  callback_moe_droprate: True
  ...

callback:
  ...
  - type: MoEDropRateCallback
    expert_num: 4
    capacity_factor: 1.5
    num_layers: 8
    mtp_depth: 1
  ...
```

#### Key Configuration Parameters

| Parameter            | Description                | Value Specification                       |
|---------------|-------------------|----------------------------|
| callback_moe_droprate | Whether to print MoE Droprate in callback. | (bool, optional) - Default: `False` .|
| expert_num | Number of experts. | (int, required) 。-  Default: `None`。 |
| capacity_factor | Capacity factor. | (float, required) 。- Default: `None`。 |
| num_layers | Number of model layers. | (int, required) 。- Default: `None`。 |
| mtp_depth | Number of MTP layers. | (int, required) 。- Default: `None`。 |

## Rotary Position Embedding Fusion Operator

### Overview

When RoPE (Rotary Position Embedding) is used as the position encoding in the network, this fusion operator can be enabled to improve overall performance. This feature provides a fused implementation of RoPE, enhancing network performance. For the operator interface, refer to:
[mindspore.ops.rotary_position_embedding](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.rotary_position_embedding.html)

### Configuration and Usage

#### YAML Parameter Configuration

To use the rotary_position_embedding fusion operator, users need to configure the `use_fused_rope` parameter under the `model_config` section in the configuration file and set it to `True`. Example:

```yaml
model_config:
  ...
  use_fused_rope: True
  ...
```

## SwiGLU Fusion Operator

### Overview

When SwiGLU is used as the activation function in the network, this fusion operator can be enabled to improve overall performance. This feature provides a fused implementation of SwiGLU, enhancing network performance. For the operator functionality, refer to:
[mindspore.ops.swiglu](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.swiglu.html).

### Configuration and Usage

#### YAML Parameter Configuration

To use the SwiGLU fusion operator, users need to configure the `use_fused_swiglu` parameter under the `model_config` section in the configuration file and set it to `True`. For example:

```yaml
model_config:
  ...
  use_fused_swiglu: True
  ...
```