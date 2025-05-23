# Other features

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/function/other_features.md)

During the large-scale training of deep learning models, challenges such as memory limitations, effective utilization of computational resources, and synchronization issues in distributed training are encountered. To address these challenges, training optimization algorithms are employed to enhance training efficiency, accelerate convergence, and improve the final model performance.

MindSpore Transformer provides optimization algorithms like Recomputation, Gradient Accumulation, and Gradient Clipping for use during training.

## Recomputation

### Overview

Recomputation can significantly reduce activation memory usage during training but at the cost of additional computations. For more information about the principles of recalculation and framework measurement capabilities, please refer to [MindSpore Tutorial Document: Recompute](https://www.mindspore.cn/tutorials/en/master/parallel/recompute.html).

### Configuration and Usage

#### YAML Parameter Configuration

Users can enable recomputation by adding a `recompute_config` module to the YAML configuration file used for model training.

Taking the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L113) as an example, it could be configured as follows:

```yaml
# recompute config
recompute_config:
  recompute: [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 0]
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True
  recompute_slice_activation: True
```

For specific configurations targeting individual layers, a tuple-based approach can be used.

For instance, with a network having 48 layers, pp_interleave_num set to 2, pipeline_stage set to 5, and offset configured as [[0,1,1,1,1],[1,1,1,1,0]], the recomputation configuration would look like this:

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

The log will print the recalculation strategy information after normalizing the input format:

```log
INFO - Formative layer_recompute: [[2, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
INFO - Formative select_recompute: {'feed_forward\.w1\.activation\.silu': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.mul': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.w1\.matmul': [[1, 0, 0, 0, 0], [2, 1, 0, 0, 0]], 'feed_forward\.w3\.matmul': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]}
INFO - Formative select_comm_recompute: {'ffn_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'attention_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]]}
```

Then the configuration of each layer recompute will be printed.

> 1. If both full recomputation and selective recomputation are configured for a layer, full recomputation takes effect.
> 2. Integers in a one-dimensional integer list or tuple can be replaced with True or False to enable or disable recomputation for all layers.

#### Key Parameters Introduction

The main parameters for recomputation configuration are listed in the following table:

| Parameter                         | Description                                                                                                            | ValueDescription                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| recompute                         | (By layer) Full recompute.                                                                                             | Can be configured as bool, list or tuple of integers, or 2D list or tuple. <br>When configured as bool type, turn on or off full recompute for all layers; <br>When configured as list or tuple of integers, it indicates how many layers in each `pipline_stage` have full recompute enabled. When `pp_interleave_num > 1`, the number of recompute layers enabled will be evenly distributed to each interleave; <br>When configured as a 2D list or tuple of integers, it indicates how many layers in each mini stage have full recompute enabled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| select_recompute                  | (By operator) Select recompute.                                                                                        | Can be configured as bool, list or tuple of integers, or two-dimensional list or tuple, list or tuple of strings, and dict. <br>The default selection recalculation operator is `['feed_forward\\.mul', 'feed_forward\\.w1\\.activation\\.silu']` . <br>When configured as bool type, it turns on or off the selection recalculation of the default operator for all layers; <br>When configured as an integer list or tuple, it represents how many layers in each `pipline_stage` turn on the selection recalculation of the default operator. When `pp_interleave_num > 1`, the number of selection recalculation layers turned on will be evenly distributed to each interleave; <br>When configured as an integer two-dimensional list or tuple, it represents how many layers in each mini stage turn on the selection recalculation of the default operator. <br>When configured as a string list or tuple, it indicates which operators are enabled for selective recomputation. The operator names are matched by regular expressions, and the hierarchical relationships are separated by `'\\.'`; <br>When configured as a dict, the key value corresponds to the operator name, and the value corresponds to the configuration method for selective recomputation. This method can fine-tune the recomputation strategy for each operator. |
| select_comm_recompute             | Select communication recomputation (by operator).                                                                      | The configuration method is the same as **select_recompute**. The default selection of communication recomputation operators is `['.*\\.norm']` . Generally, it is only configured for layer_norm or similar layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| parallel_optimizer_comm_recompute | Optimizer parallel communication recomputation. Whether to recompute AllGather communication in optimizer parallelism. | (bool, optional) - After enabling, in automatic parallelism or semi-automatic parallelism mode, specify whether AllGather communication introduced by optimizer parallelism in Cell is recomputed. Default value: `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| mp_comm_recompute                 | Model parallel communication recomputation, whether to recompute communication operators in model parallelism.         | (bool, optional) - After turning on, in automatic parallelism or semi-automatic parallelism mode, specify whether to recompute the communication operations introduced by model parallelism in the cell. Default value: `True`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| recompute_slice_activation        | Slice recomputation, whether to slice the cell output that will be kept in memory.                                     | (bool, optional) - Default value: `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

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