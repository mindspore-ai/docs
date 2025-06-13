# 其它训练特性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/other_training_features.md)

在大规模的深度学习模型训练中，会遇到诸如：内存限制、计算资源的有效利用、分布式训练中的同步问题等挑战，需要使用训练优化算法来提高训练效率、加速收敛速度以及改善最终模型性能。

MindSpore Transformers 提供了梯度累积、梯度裁剪等训练优化算法，可供开发者进行训练时使用。

## 梯度累积

### 概述

MindSpore 在 2.1.1 之后的版本中增加了 `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` 这一梯度累积实现接口，通过拆分 MiniBatch 的形式提供了梯度累加的能力，MindSpore Transformers 将其封装进了统一的训练流程，通过 yaml 配置进行使能。关于梯度累积的原理和框架测的能力可以参考 [MindSpore 文档：梯度累加](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/distributed_gradient_accumulation.html)。

### 配置与使用

#### YAML 参数配置

用户在需要开启梯度累积的场景下，只需在配置文件中的 `runner_config` 项下配置 `gradient_accumulation_steps` 项，设置为所需的梯度累积步数即可：

```yaml
# runner config
runner_config:
  ...
  gradient_accumulation_steps: 4
  ...
```

#### 主要配置参数介绍

| 参数                          | 描述                              | 取值说明                   |
|-----------------------------|---------------------------------|------------------------|
| gradient_accumulation_steps | 在执行反向传播前，累积梯度的步数。 | (int, 必选) - 默认值： `1` 。 |

#### 其他方式使用梯度累积

除配置文件外，当采用 `run_mindformer.py` 脚本启动时，可指定 `--gradient_accumulation_steps` 入参来使用梯度累积功能。

#### 梯度累积使用限制

> 开启梯度累积会增大内存开销，请注意内存管理，防止发生内存溢出（OOM）。

1. 由于 `GradAccumulationCell` 的实现依赖并行特性，梯度累积当前仅支持在**半自动并行模式**下使用；
2. 此外，在 pipeline 并行场景下，梯度累积含义与 micro_batch 相同，将不会生效，请配置 `micro_batch_num` 项以增大训练 batch_size。

## 梯度裁剪

### 概述

梯度裁剪算法可以避免反向梯度过大，跳过最优解的情况。

### 配置与使用

#### YAML 参数配置

在 MindSpore Transformers 中，默认的训练流程 `MFTrainOneStepCell` 中集成了梯度裁剪逻辑。

可使用如下示例，以开启梯度裁剪：

```yaml
# wrapper cell config
runner_wrapper:
  type: MFTrainOneStepCell
  ...
  use_clip_grad: True
  max_grad_norm: 1.0
  ...
```

#### 主要配置参数介绍

| 参数            | 描述                | 取值说明                       |
|---------------|-------------------|----------------------------|
| use_clip_grad | 控制在训练过程中是否开启梯度裁剪。 | (bool, 可选) - 默认值： `False` 。 |
| max_grad_norm | 控制梯度裁剪的最大 norm 值。 | (float, 可选) - 默认值： `1.0` 。 |

## GroupedMatmul

### 概述

针对MoE单卡多专家计算，存在细碎的专家计算操作与通信，通过GroupedMatmul算子对多专家计算进行合并，提升MoE单卡多专家训练性能。通过调用GroupedMatmul算子，对多个专家计算进行融合达到加速效果。

### 配置与使用

#### YAML 参数配置

用户在需要MoE开启GroupedMatmul的场景下，只需在配置文件中的 `moe_config` 项下配置 `use_gmm` 项，设置为`True`即可：

```yaml
moe_config:
  ...
  use_gmm: True
  ...
```

### FAQ

使用GroupedMatmul融合算子，在负载不均衡时可能会出现某张卡上的专家未被分配任何token的情况，导致程序报错。报错如下：

```log
VallueError: For primitive[Reshape]， the accumulate of x_shape must be equal to out_shape, but got x_shape: [const vector]{}, and output_shape: [const vector]{0, hiddensize}
```

此时，可以配置enable_gmm_safe_tokens: True，保证每个专家至少分配1个tokens，避免程序报错。

```yaml
moe_config:
  ...
  enable_gmm_safe_tokens: True
  ...
```

## MoE Droprate打印

### 概述

在使用MoE（Mixture of Experts）容量方案进行模型训练时，为了提高效率和性能，系统可能会对某些token执行drop操作。通过启用droprate打印功能，用户可以在训练过程中实时监控这些drop操作的发生率，从而更好地理解模型的行为，并据此调整训练策略。此功能允许用户在训练过程中查看每一层的droprate情况。droprate是指在特定层中被drop掉的token的比例。通过观察droprate的变化趋势，可以帮助用户评估当前的训练参数设置是否合理，以及模型是否有效地利用了专家资源。

### 配置与使用

#### YAML 参数配置

用户要启用droprate打印功能，需在配置文件中的 `moe_config` 项下配置 `callback_moe_droprate` 项，设置为`True`，在callback部分添加`MoEDropRateCallback`配置项，并设置模型相关参数`expert_num`、`capacity_factor`、`num_layers`、`mtp_depth`。示例：

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

#### 主要配置参数介绍

| 参数            | 描述                | 取值说明                       |
|---------------|-------------------|----------------------------|
| callback_moe_droprate | 是否在callback中打印MoE Droprate。 | (bool, 可选) - 默认值： `False` 。 |
| expert_num | 专家数量。 | (int, 必选) - 默认值： `None`。 |
| capacity_factor | 容量因子。 | (float, 必选) - 默认值： `None`。 |
| num_layers | 模型层数。 | (int, 必选) - 默认值： `None`。 |
| mtp_depth | mtp层层数。 | (int, 必选) - 默认值： `None`。 |

## RoPE融合算子

### 概述

网络中使用RoPE（Rotary Position Embedding）作为位置编码时，可以启用该融合算子提升整网性能。该功能提供RoPE的融合算子实现，提升整网性能。算子的接口可参考：
[mindspore.ops.rotary_position_embedding](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.rotary_position_embedding.html)。

### 配置与使用

#### YAML 参数配置

用户需要使用rotary_position_embedding融合算子，需在配置文件中的 `model_config` 项下配置 `use_fused_rope` 项，设置为`True`，示例：

```yaml
model_config:
  ...
  use_fused_rope: True
  ...
```

## SwiGLU融合算子

### 概述

网络中使用SwiGLU作为激活函数时可以启用该融合算子提升整网性能。该功能提供SwiGLU的融合算子实现，提升整网性能。算子的功能可参考：
[mindspore.ops.swiglu](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.swiglu.html)。

### 配置与使用

#### YAML 参数配置

用户需要使用SwiGLU融合算子，需在配置文件中的 `model_config` 项下配置 `use_fused_swiglu` 项，设置为`True`，示例：

```yaml
model_config:
  ...
  use_fused_swiglu: True
  ...
```
