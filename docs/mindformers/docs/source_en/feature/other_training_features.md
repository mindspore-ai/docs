# Other Training Features

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/other_training_features.md)

During the large-scale training of deep learning models, challenges such as memory limitations, effective utilization of computational resources, and synchronization issues in distributed training are encountered. To address these challenges, training optimization algorithms are employed to enhance training efficiency, accelerate convergence, and improve the final model performance.

MindSpore Transformer provides optimization algorithms like Recomputation, Gradient Accumulation, and Gradient Clipping for use during training.

## Gradient Accumulation

### Overview

MindSpore supported the gradient accumulation implementation interface `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` in versions after 2.1.1, which provides the gradient accumulation capability by splitting MiniBatch. MindSpore Transformer encapsulates it into a unified training process and enables it through yaml configuration. For the principle of gradient accumulation and the ability of framework measurement, please refer to [MindSpore Document: Gradient Accumulation](https://www.mindspore.cn/tutorials/en/master/parallel/distributed_gradient_accumulation.html).

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

In MindSpore TransFormers, the default training process `MFTrainOneStepCell` integrates gradient clipping logic.

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