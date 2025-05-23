# Model Training Hyperparameters Configuration

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/function/training_hyperparameters.md)

## Overview

Hyperparameters significantly affect model performance, with different settings potentially leading to vastly different outcomes.

Choices regarding these parameters influence aspects such as training speed, convergence, capacity, and generalization ability. They are not learned directly from the training data but are determined by developers based on experience, experiments, or tuning processes.

MindSpore Transformer offers several categories of hyperparameter configuration methods.

## Learning Rate

### Overview

The learning rate controls the size of the step taken during updates to model weights, determining the pace at which parameters are updated.

It is a critical parameter affecting both the training speed and stability of the model. During each iteration, gradients of the loss function with respect to the weights are calculated and adjusted according to the learning rate.

Setting the learning rate too high can prevent the model from converging, while setting it too low can make the training process unnecessarily slow.

### Configuration and Usage

#### YAML Parameter Configuration

Users can utilize the learning rate by adding an `lr_schedule` module to the YAML configuration file used for model training.

Taking the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L31) as an example, it could be configured as follows:

```yaml
# lr schedule
lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 2.2e-4
  warmup_ratio: 0.02
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

#### Key Parameters Introduction

Different learning rates require different configuration parameters. MindSpore Transformer currently supports the following learning rates:

1. [Constant Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.ConstantWarmUpLR.html)
2. [Linear with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.LinearWithWarmUpLR.html)
3. [Cosine with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.CosineWithWarmUpLR.html)
4. [Cosine with Restarts and Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.CosineWithRestartsAndWarmUpLR.html)
5. [Polynomial with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.PolynomialWithWarmUpLR.html)
6. [The cosine annealing part of SGDR](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.CosineAnnealingLR.html)
7. [Set the learning rate of each parameter group using a cosine annealing schedule](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.CosineAnnealingWarmRestarts.html)
8. [Learning Rate Wise Layer](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.LearningRateWiseLayer.html)

Taking the cosine warm-up learning rate (CosineWithWarmUpLR) as an example, the main parameters that need to be paid attention to are listed in the following table:

| Parameter      | Description                                    | Value Description                                                        |
|----------------|------------------------------------------------|--------------------------------------------------------------------------|
| type           | Type of learning rate to use.                  | (str, required) - Such as `ConstantWarmUpLR`, `CosineWithWarmUpLR`, etc. |
| learning_rate  | Initial value of learning rate.                | (float, required) - Default value: `None`.                               |
| warmup_steps   | Number of steps in the warmup phase.           | (int, optional) - Default value: `None`.                                 |
| warmup_lr_init | Initial learning rate in the warmup phase.     | (float, optional) - Default value: `0.0`.                                |
| warmup_ratio   | Ratio of warmup phase to total training steps. | (float, optional) - Default value: `None`.                               |
| total_steps    | Total number of warmup steps.                  | (int, optional) - Default value: `None`.                                 |
| lr_end         | Final value of the learning rate.              | (float, optional) - Default value: `0.0`.                                |

In yaml file, the following configuration can be made, indicating that the cosine warmup learning rate with an initial value of 1e-5 is used, the total warmup steps are 20, and the warmup phase accounts for 1% of the total training steps:

```yaml
# lr schedule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1e-5
  warmup_lr_init: 0.0
  warmup_ratio: 0.01
  warmup_steps: 0
  total_steps: 20 # -1 means it will load the total steps of the dataset
```

For more details about the learning rate API (such as `type` configuration names and introductions to learning rate algorithms), please refer to the related links in the [MindSpore TransFormer API Documentation: Learning Rate](https://www.mindspore.cn/mindformers/docs/en/dev/mindformers.core.html#learning-rate).

## Optimizer

### Overview

An optimizer is an algorithmic choice used for optimizing neural network weights during training by updating model weights to minimize the loss function.

Selecting the right optimizer is crucial for the convergence speed and final performance of the model. Different optimizers employ various strategies to adjust the learning rate and other hyperparameters to accelerate the training process, improve convergence, and avoid local optima.

Currently, MindSpore Transformer only supports the [AdamW optimizer](https://www.mindspore.cn/mindformers/docs/en/dev/mindformers.core.html#optimizer).

### Configuration and Usage

#### YAML Parameter Configuration

Users can use the optimizer by adding an `optimizer` module to the YAML configuration file for model training.

Taking the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L24) as an example, it could be configured like this:

```yaml
# optimizer
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
```

#### Key Parameters Introduction

For the main parameters of optimizer configuration, see the relevant link in [MindSpore TransFormer API Documentation: Optimizer](https://www.mindspore.cn/mindformers/docs/en/dev/core/mindformers.core.AdamW.html#mindformers.core.AdamW).