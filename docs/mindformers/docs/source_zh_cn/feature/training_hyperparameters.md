# 模型训练超参数配置

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/training_hyperparameters.md)

超参数对模型的性能有着重要影响，不同的超参数设置可能导致模型表现的巨大差异。参数的选择会影响到模型的训练速度、收敛性、容量和泛化能力等方面。且它们并非通过训练数据直接学习得到的，而是由开发者根据经验、实验或调优过程来确定的。

MindSpore Transformers 提供了如下几类超参数的配置方式。

## 学习率

### 概述

学习率控制着模型权重更新的步长大小，决定了参数更新的速度。

学习率是影响模型训练速度和稳定性的关键参数。在每次迭代过程中，通过计算损失函数相对于权重的梯度，并根据学习率调整这些权重。学习率设置得过大可能会导致模型无法收敛，而设置得过小则会使训练过程过于缓慢。

### 配置与使用

#### YAML 参数配置

用户可通过在模型训练的 yaml 配置文件中新增 `lr_schedule` 模块来使用学习率。
以 [`DeepSeek-V3` 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L31) 为例，可做如下配置：

```yaml
# lr schedule
lr_schedule:
  type: ConstantWarmUpLR
  learning_rate: 2.2e-4
  warmup_ratio: 0.02
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

#### 主要配置参数介绍

各学习率需配置的参数不同，MindSpore Transformers 目前支持了以下学习率：

1. [恒定预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.ConstantWarmUpLR.html)
2. [线性预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.LinearWithWarmUpLR.html)
3. [余弦预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CosineWithWarmUpLR.html)
4. [余弦重启与预热学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CosineWithRestartsAndWarmUpLR.html)
5. [带有预热阶段的多项式衰减学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.PolynomialWithWarmUpLR.html)
6. [SGDR 的余弦退火部分](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CosineAnnealingLR.html)
7. [使用余弦退火调度设置每个参数组的学习率](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CosineAnnealingWarmRestarts.html)
8. [学习率分层模块](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.LearningRateWiseLayer.html)

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

更多关于学习率 API 的介绍（如 `type` 的配置名称、学习率算法的介绍），可参见 [MindSpore Transformers API 文档：学习率部分](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/mindformers.core.html#%E5%AD%A6%E4%B9%A0%E7%8E%87) 的相关链接。

## 优化器

### 概述

优化器是用于优化神经网络权重的算法选择，其在训练过程中更新模型权重以最小化损失函数。

选择合适的优化器对模型的收敛速度和最终性能有着至关重要的影响。不同的优化器通过不同的方法调整学习率和其他超参数来加速训练过程、改善收敛性并避免局部最优解。

当前，MindSpore Transformers 只支持 [AdamW 优化器](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/mindformers.core.html#%E4%BC%98%E5%8C%96%E5%99%A8)。

### 配置与使用

#### YAML 参数配置

用户可通过在模型训练的 yaml 配置文件中新增 `optimizer` 模块来使用学习率。

以 [DeepSeek-V3 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L24) 为例，可做如下配置：

```yaml
# optimizer
optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
```

#### 主要配置参数介绍

有关优化器配置的主要参数，可参见 [MindSpore Transformers API 文档：优化器部分](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.AdamW.html#mindformers.core.AdamW) 的相关链接。
