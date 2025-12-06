# Training Hyperparameters

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/training_hyperparameters.md)

Hyperparameters significantly affect model performance, with different settings potentially leading to vastly different outcomes.

Choices regarding these parameters influence aspects such as training speed, convergence, capacity, and generalization ability. They are not learned directly from the training data but are determined by developers based on experience, experiments, or tuning processes.

MindSpore Transformers offers several categories of hyperparameter configuration methods.

## Learning Rate

### Dynamic Learning Rate

The learning rate controls the size of the step taken during updates to model weights, determining the pace at which parameters are updated.

It is a critical parameter affecting both the training speed and stability of the model. During each iteration, gradients of the loss function with respect to the weights are calculated and adjusted according to the learning rate.

Setting the learning rate too high can prevent the model from converging, while setting it too low can make the training process unnecessarily slow.

**YAML Parameter Configuration**

Users can utilize the learning rate by adding an `lr_schedule` module to the YAML configuration file used for model training.

Taking the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/deepseek3/pretrain_deepseek3_671b.yaml) as an example, it could be configured as follows:

```yaml
# lr schedule
lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 2.2e-4
  warmup_ratio: 0.02
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

**Key Parameters Introduction**

Different learning rates require different configuration parameters. MindSpore Transformers currently supports the following learning rates:

1. [Constant Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.ConstantWarmUpLR.html)
2. [Linear with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.LinearWithWarmUpLR.html)
3. [Cosine with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.CosineWithWarmUpLR.html)
4. [Cosine with Restarts and Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.CosineWithRestartsAndWarmUpLR.html)
5. [Polynomial with Warm Up Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.PolynomialWithWarmUpLR.html)
6. [The cosine annealing part of SGDR](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.CosineAnnealingLR.html)
7. [Set the learning rate of each parameter group using a cosine annealing schedule](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.CosineAnnealingWarmRestarts.html)
8. [Learning Rate Wise Layer](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.LearningRateWiseLayer.html)

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

For more details about the learning rate API (such as `type` configuration names and introductions to learning rate algorithms), please refer to the related links in the [MindSpore Transformers API Documentation: Learning Rate](https://www.mindspore.cn/mindformers/docs/en/master/mindformers.core.html#learning-rate).

### Grouped Learning Rate

Since different layers or parameters in a model have varying sensitivities to the learning rate, configuring different learning rate strategies for different parameters during training can improve training efficiency and performance. This helps avoid overfitting or insufficient training in certain parts of the network.

To enable grouped learning rate functionality, configure the `grouped_lr_schedule` field in the configuration file. This configuration includes two configurable options: `default` and `grouped`.

| Parameter   | Description                                                                                                                                                        | Type  |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| default     | The learning rate strategy for parameters that do not require grouping. The configuration contents are the same as the `lr_schedule` in [Dynamic Learning Rate](#dynamic-learning-rate). | dict  |
| grouped     | Each parameter group and its corresponding learning rate strategy configuration. Compared to the `lr_schedule` in [Dynamic Learning Rate] (#dynamic-learning-rate), an additional `params` parameter needs to be configured for each parameter group. The model's parameters are matched using regex, and the corresponding learning rate strategy is applied. | list  |

> When both lr_schedule and grouped_lr_schedule are set, lr_schedule will not take effect.

Here is an example of grouped learning rate configuration:

```yaml
grouped_lr_schedule:
  default:
    type: LinearWithWarmUpLR
    learning_rate: 5.e-5
    warmup_steps: 0
    total_steps: -1 # -1 means it will load the total steps of the dataset
  grouped:
    - type: LinearWithWarmUpLR
      params: ['embedding.*', 'output_layer.weight']
      learning_rate: 2.5e-5
      warmup_steps: 0
      total_steps: -1
    - type: ConstantWarmUpLR
      params: ['q_layernorm', 'kv_layernorm']
      learning_rate: 5.e-6
      warmup_steps: 0
      total_steps: -1
```

## Optimizer

### Overview

An optimizer is an algorithmic choice used for optimizing neural network weights during training by updating model weights to minimize the loss function.

Selecting the right optimizer is crucial for the convergence speed and final performance of the model. Different optimizers employ various strategies to adjust the learning rate and other hyperparameters to accelerate the training process, improve convergence, and avoid local optima.

MindSpore Transformers currently supports the following optimizers:

- [**AdamW Optimizer**](https://www.mindspore.cn/mindformers/docs/en/master/mindformers.core.html#optimizer)
- **Muon Optimizer**

These optimizers use different mathematical strategies—such as adaptive learning rates, momentum estimation, and direction normalization—to influence training stability, convergence characteristics, and final accuracy.

Users can use the optimizer by adding an `optimizer` module to the YAML configuration file for model training.

The following example is based on the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/deepseek3/pretrain_deepseek3_671b.yaml).

### AdamW Optimizer

AdamW is an optimizer based on Adaptive Moment Estimation (Adam) with an improved decoupled weight decay formulation. It maintains first-order and second-order moment estimates of gradients to provide adaptive learning rates, enabling stable and efficient parameter updates during training.

Thanks to its robustness and strong convergence behavior, AdamW is widely used in large-scale Transformer models, LLM pretraining, and architectures such as MoE. It remains one of the most commonly applied optimizers in modern deep learning systems.

#### YAML Example

```yaml
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01
```

#### Key Parameters Introduction

For the main parameters of optimizer configuration, see the relevant link in [MindSpore Transformers API Documentation: Optimizer](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.AdamW.html#mindformers.core.AdamW).

### Muon Optimizer

Muon (Momentum Orthogonalized by Newton-Schulz) is a matrix-structured and geometry-aware optimizer designed for large-scale deep learning, especially LLM training. It optimizes 2D neural network parameters by first taking the updates produced by SGD with momentum. Then, it applies a Newton–Schulz iteration as a post-processing step to each update before applying it to the parameters. For details, see [Muon Optimizer Documentation](https://kellerjordan.github.io/posts/muon/).

#### YAML Example

```yaml
optimizer:
  type: Muon
  adamw_betas: [0.9, 0.95]
  adamw_eps: 1.e-8
  weight_decay: 0.01
  matched_adamw_rms: 0.2
  qk_clip_threshold: 100
```

#### Key Parameters Introduction

- `adamw_betas` (list[float] or tuple[float], optional): Exponential decay rates for the first and second moment estimates, used to match AdamW’s momentum statistics. Each value must lie within (0.0, 1.0). Default: (0.95, 0.95).
- `adamw_eps` (float, optional): A small constant added to the denominator to improve numerical stability. Must be greater than 0. Default: 1e-8.
- `weight_decay` (float, optional): The coefficient for L2 weight decay, used to regularize parameters during optimization. Default: 0.1.
- `matched_adamw_rms` (float, optional): Matches the RMS (root-mean-square) magnitude of AdamW updates to ensure compatible update scales—preventing instability from overly large steps and avoiding slow convergence from overly small steps. Default: 0.2.
- `qk_clip_threshold` (float, optional): A clipping threshold applied to Q/K dot-product attention scores to prevent excessively large softmax inputs, which can cause numerical instability or gradient explosions. Default: 100.
