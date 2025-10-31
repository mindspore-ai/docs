# 训练超参数

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/training_hyperparameters.md)

超参数对模型的性能有着重要影响，不同的超参数设置可能导致模型表现的巨大差异。参数的选择会影响到模型的训练速度、收敛性、容量和泛化能力等方面。且它们并非通过训练数据直接学习得到的，而是由开发者根据经验、实验或调优过程来确定的。

MindSpore Transformers 提供了如下几类超参数的配置方式。

## 学习率

### 动态学习率

学习率控制着模型权重更新的步长大小，决定了参数更新的速度。

学习率是影响模型训练速度和稳定性的关键参数。在每次迭代过程中，通过计算损失函数相对于权重的梯度，并根据学习率调整这些权重。学习率设置得过大可能会导致模型无法收敛，而设置得过小则会使训练过程过于缓慢。

**YAML 参数配置**

用户可通过在模型训练的 yaml 配置文件中新增 `lr_schedule` 模块来使用学习率。
以 [`DeepSeek-V3` 预训练 yaml](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/deepseek3/pretrain_deepseek3_671b.yaml) 为例，可做如下配置：

```yaml
# lr schedule
lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 2.2e-4
  warmup_ratio: 0.02
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

**主要配置参数介绍**

各学习率需配置的参数不同，MindSpore Transformers 目前支持了以下学习率：

1. [恒定预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.ConstantWarmUpLR.html)
2. [线性预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.LinearWithWarmUpLR.html)
3. [余弦预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.CosineWithWarmUpLR.html)
4. [余弦重启与预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.CosineWithRestartsAndWarmUpLR.html)
5. [带有预热阶段的多项式衰减学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.PolynomialWithWarmUpLR.html)
6. [SGDR 的余弦退火部分](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.CosineAnnealingLR.html)
7. [使用余弦退火调度设置每个参数组的学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.CosineAnnealingWarmRestarts.html)
8. [学习率分层模块](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.LearningRateWiseLayer.html)

以余弦预热学习率（CosineWithWarmUpLR）为例，需要关注的主要参数如下表所列：

| 参数             | 描述             | 取值说明                                                       |
|----------------|----------------|------------------------------------------------------------|
| type           | 使用学习率的类型。      | (str, 必选) - 如 `ConstantWarmUpLR` 、 `CosineWithWarmUpLR` 等。 |
| learning_rate  | 学习率的初始值。       | (float, 必选) - 默认值： `None` 。                                |
| warmup_steps   | 预热阶段的步数。       | (int, 可选) - 默认值： `None` 。                                  |
| warmup_lr_init | 预热阶段的初始学习率。    | (float, 可选) - 默认值： `0.0` 。                                 |
| warmup_ratio   | 预热阶段占总训练步数的比例。 | (float, 可选) - 默认值： `None` 。                                |
| total_steps    | 总的预热步数。        | (int, 可选) - 默认值： `None` 。                                  |
| lr_end         | 学习率的最终值。       | (float, 可选) - 默认值： `0.0` 。                                 |

在 yaml 中，可做如下配置，表示使用初始值为 1e-5 的余弦预热学习率，总预热 20 步，预热阶段占总训练步数的 1%：

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

更多关于学习率 API 的介绍（如 `type` 的配置名称、学习率算法的介绍），可参见 [MindSpore Transformers API 文档：学习率部分](https://www.mindspore.cn/mindformers/docs/zh-CN/master/mindformers.core.html#%E5%AD%A6%E4%B9%A0%E7%8E%87) 的相关链接。

### 分组学习率

由于模型中不同层或参数对学习率的敏感度不同，在训练过程中针对不同的参数设置不同的学习率策略能够提高训练效率和性能，避免网络中部分参数过拟合训练或训练不充分的情况发生。

在配置文件中配置`grouped_lr_schedule`字段即可开启分组学习率功能，该配置下包含`default`和`grouped`两个可配置项：

| 参数名     | 说明                                                                                                                                             | 类型   |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------|------|
| default | 不需要分组的参数对应的学习率策略配置，可配置内容与[动态学习率](#动态学习率)中`lr_schedule`相同。                                                                                       | dict |
| grouped | 各参数组及其对应的学习率策略配置，每个参数组中可配置内容与[动态学习率](#动态学习率)中`lr_schedule`相比需要额外配置`params`参数； <br/> `params`是一个字符串列表，表示需要匹配的参数名，配置后会通过正则匹配模型中的参数名并配置对应的学习率策略。 | list |

> 当同时配置lr_schedule和grouped_lr_schedule时，lr_schedule不生效。

以下是分组学习率配置示例：

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

## 优化器

### 概述

优化器是用于优化神经网络权重的算法选择，其在训练过程中更新模型权重以最小化损失函数。

选择合适的优化器对模型的收敛速度和最终性能有着至关重要的影响。不同的优化器通过不同的方法调整学习率和其他超参数来加速训练过程、改善收敛性并避免局部最优解。

当前，MindSpore Transformers 只支持 [AdamW 优化器](https://www.mindspore.cn/mindformers/docs/zh-CN/master/mindformers.core.html#%E4%BC%98%E5%8C%96%E5%99%A8)。

### 配置与使用

#### YAML 参数配置

用户可通过在模型训练的 yaml 配置文件中新增 `optimizer` 模块来使用学习率。

以 [DeepSeek-V3 预训练 yaml](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/deepseek3/pretrain_deepseek3_671b.yaml) 为例，可做如下配置：

```yaml
# optimizer
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
```

#### 主要配置参数介绍

有关优化器配置的主要参数，可参见 [MindSpore Transformers API 文档：优化器部分](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.AdamW.html#mindformers.core.AdamW) 的相关链接。
