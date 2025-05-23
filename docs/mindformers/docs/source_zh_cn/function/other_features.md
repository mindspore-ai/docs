# 其它特性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/other_features.md)

在大规模的深度学习模型训练中，会遇到诸如：内存限制、计算资源的有效利用、分布式训练中的同步问题等挑战，需要使用训练优化算法来提高训练效率、加速收敛速度以及改善最终模型性能。

MindSpore TransFormer 提供了重计算、梯度累积、梯度裁剪等训练优化算法，可供开发者进行训练时使用。

## 重计算

### 概述

重计算可以显著降低训练时的激活内存，但会额外增加一些计算。关于重计算的原理和框架测能力可参考 [MindSpore 教程文档：重计算](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/recompute.html) 。

### 配置与使用

#### YAML 参数配置

用户可通过在模型训练的 yaml 配置文件中新增 `recompute_config` 模块来使用重计算。

以 [DeepSeek-V3 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L113) 为例，可做如下配置：

```yaml
# recompute config
recompute_config:
  recompute: [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 0]
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True
  recompute_slice_activation: True
```

如果需要对选择重计算配置到某几个特定层进行，可以使用 tuple 的方式进行配置。

例如：一个网络有48层， `pp_interleave_num` 为 `2` ， `pipeline_stage` 为 `5` ，offset设为 `[[0,1,1,1,1],[1,1,1,1,0]]` ，重计算配置如下：

```yaml
# recompute config
recompute_config:
  recompute: [[2,1,0,0,0],[1,0,0,0,0]]
  select_recompute:
    'feed_forward\.w1\.activation\.silu': True
    'feed_forward\.mul': True
    'feed_forward\.w1\.matmul': [[1,0,0,0,0],[2,1,0,0,0]]
    'feed_forward\.w3\.matmul': [2,1,0,0,0]
  select_comm_recompute: ['ffn_norm\.norm','attention_norm\.norm']
```

在日志中会打印将输入格式规范化后的重计算策略信息：

```log
INFO - Formative layer_recompute: [[2, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
INFO - Formative select_recompute: {'feed_forward\.w1\.activation\.silu': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.mul': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.w1\.matmul': [[1, 0, 0, 0, 0], [2, 1, 0, 0, 0]], 'feed_forward\.w3\.matmul': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]}
INFO - Formative select_comm_recompute: {'ffn_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'attention_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]]}
```

随后会打印每一层重计算的配置方式。

> 注：
> 1. 如果某一层同时配置了完全重计算与选择重计算，则按完全重计算生效。
> 2. 在一维整数型 list 或 tuple 中的整数可以替换为 True 或 False，代表对所有层启用或关闭重计算。

#### 主要配置参数介绍

有关重计算配置的主要参数如下表所列：

| 参数                                | 描述                                                       | 取值说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| recompute                         | （按层）完全重计算。                                               | 可配置为 bool，整数型的 list 或 tuple，或二维 list 或 tuple。<br>配置为 bool 类型时，对所有层开启或关闭完全重计算；<br>配置为整数型 list 或 tuple 时，代表每个 `pipline_stage` 中有多少层开启完全重计算， `pp_interleave_num > 1` 时开启的重计算层数会均匀分配到各 interleave 中；<br>配置为整数型二维 list 或 tuple 时，代表每个 mini stage 中有多少层开启完全重计算。                                                                                                                                                                                                                                                                         |
| select_recompute                  | （按算子）选择重计算。                                              | 可配置为 bool，整数型的 list 或 tuple，或二维 list 或 tuple，字符串的 list 或 tuple，以及 dict。<br>默认选择重计算算子为 `['feed_forward\\.mul', 'feed_forward\\.w1\\.activation\\.silu']` 。<br>配置为 bool 类型时，对所有层开启或关闭默认算子的选择重计算；<br>配置为整数型 list 或 tuple 时，代表每个 `pipline_stage` 中有多少层开启默认算子的选择重计算， `pp_interleave_num > 1` 时开启的选择重计算层数会均匀分配到各 interleave 中；<br>配置为整数型二维 list 或 tuple 时，代表每个 mini stage 中有多少层开启默认算子的选择重计算。<br>配置为字符串 list 或 tuple 时，代表对哪些算子开启选择重计算，算子名通过正则表达式匹配，层级关系通过 `'\\.'` 分割；<br>配置为 dict 时，key 值对应算子名，value 值对应选择重计算的配置方式，这种配法可以对每个算子精细配置重计算策略。 |
| select_comm_recompute             | （按算子）选择通信重计算。                                            | 配置方式与 **select_recompute** 相同，默认选择通信重计算算子为 `['.*\\.norm']` 。一般仅对 layer_norm 或类似层进行配置。                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| parallel_optimizer_comm_recompute | 优化器并行通信重计算。在优化器并行下，是否重计算 AllGather 通信。                   | (bool, 可选) - 开启后在自动并行或半自动并行模式下，指定 Cell 内部由优化器并行引入的 AllGather 通信是否重计算。 默认值： `False` 。                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| mp_comm_recompute                 | 模型并行通信重计算，在模型并行下，是否重计算通信算子。 | (bool, 可选) - 开启后在自动并行或半自动并行模式下，指定 Cell 内部由模型并行引入的通信操作是否重计算。默认值： `True` 。                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| recompute_slice_activation        | 切片重计算，是否对将保留在内存中的 Cell 输出进行切片。                           | (bool, 可选) - 默认值： `False` 。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

## 梯度累积

### 概述

MindSpore 在 2.1.1 之后的版本中增加了 `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` 这一梯度累积实现接口，通过拆分 MiniBatch 的形式提供了梯度累加的能力，MindSpore Transformer 将其封装进了统一的训练流程，通过 yaml 配置进行使能。关于梯度累积的原理和框架测的能力可以参考 [MindSpore 文档：梯度累加](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/distributed_gradient_accumulation.html)。

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

在 MindSpore TransFormers 中，默认的训练流程 `MFTrainOneStepCell` 中集成了梯度裁剪逻辑。

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
