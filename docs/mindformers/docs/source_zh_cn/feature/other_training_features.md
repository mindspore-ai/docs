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
