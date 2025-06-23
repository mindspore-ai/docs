# 数据跳过和健康监测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/skip_data_and_ckpt_health_monitor.md)

## 概述

数据跳过功能是指当训练过程中，遇到某个step的global norm超过设定的阈值时，会跳过当前步数训练数据；当连续累计的越界次数达到阈值时，便会触发异常中断，终止训练。而健康监测功能是指在保存权重时，对保存的权重的健康状况进行监测，生成一个文件记录权重的健康状况，并在下次续训时通过该文件来选择最新的健康的权重进行续训。

权重的健康状况判定请参考[权重健康监测](#权重健康监测)。

> - 数据跳过功能和健康监测功能二者结合，能有效解决训练过程中异常 global norm 带来的数据异常问题。使用前请先正常训练一段时间，从而确定需要设定的 global norm 的阈值、连续异常次数的阈值以及 embedding norm 的阈值。
> - 只有连续出现异常时才会中断训练，如果中途出现一次恢复正常，则会清空累计次数，所以请把控阈值的设定。
> - 数据跳过功能不能与故障快速恢复功能同时使用。参考[高可用特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/high_availability.html)中的进程级重调度恢复功能。

## 数据跳过

### 概述

MindSpore Transformers提供了跳过数据的功能，能够在global norm异常时跳过当前训练的数据，并当连续异常次数达到设定阈值时触发异常中断。

本功能一共有以下三种行为：

- 出现越界global norm，异常连续累计次数+1，跳过当前步数训练数据，打印日志信息。
- global norm恢复正常，异常连续累计次数清空。
- 异常连续累计次数达到设定阈值，触发异常中断，终止训练。

#### 使用方法

**注意**：以下示例所展示的参数数值仅作为实验数据，请以真实训练数据为准。

本功能通过YAML配置文件使能：

```yaml
use_skip_data_by_global_norm: True

monitor_config:
  monitor_on: True
  check_for_global_norm: False
  global_norm_spike_threshold: 3.0
  global_norm_spike_count_threshold: 2
```

**参数说明：**

| 参数名称                              | 描述                                                  | 类型    | 是否可选 | 取值范围 |
|-----------------------------------|-----------------------------------------------------|-------|------|------|
| use_skip_data_by_global_norm      | 数据跳过功能开关。默认值为`False`。                               | bool  | 可选   |      |
| monitor_config                    | 训练指标监控配置。默认值为`None`。                                |       | 可选    |      |
| monitor_on                        | 是否开启训练指标监控配置。默认值为`False`。                           | bool  | 可选    |      |
| check_for_global_norm             | 是否开启故障快速恢复功能，和数据跳过功能互斥。默认值为`False`。                   | bool  | 可选    |      |
| global_norm_spike_threshold       | global norm的阈值，当global norm超过时触发数据跳过。默认值为`3.0`。     | float | 可选    | 大于0  |
| global_norm_spike_count_threshold | 连续异常global norm累计的次数，当次数达到该阈值则触发异常中断，终止训练。默认值为`10`。 | int   | 可选    | 正整数  |

### 使用示例

假设以Llama3.1-8B为例子，使用的[finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml)按照上述[配置](#使用方法)添加参数，其余步骤请参考[Llama3.1-8B文档](https://gitee.com/lanshaozuishuai/mindformers/blob/dev/research/llama3_1/README.md)。开启训练：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/llama3_1 \
    --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
    --train_data /{path}/wiki4096.mindrecord \
    --run_mode train \
    --use_parallel True" 8
```

模型正式开始训练时，global norm大于设定阈值，则会打印如下日志，提示用户当前已经连续n次出现异常global norm，并跳过当前步数的训练数据。

```log
- INFO - { Epoch:[  1/  2], step:[    1/ 6500], loss: 0.000, per_step_time: 166756ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [44.313248], train_throughput_per_npu: 2.849T
- INFO -    0.0% |                                                  | 0.00600 samples/s/p  25 days, 2:07:47 }
- INFO - opt_global_step: 0, skip_data_grad_norm_threshold: 3.0, is_skip: [ True]
- INFO - Current global norm [44.313248] of step 1 has been 1 consecutive times greater than threshold: 3.0
```

当连续异常次数达到设定的阈值时，打印错误日志，终止训练。

```log
- INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 0.000, per_step_time: 7637ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [47.329006], train_throughput_per_npu: 62.211T
- INFO -    0.0% |                                                  | 0.00600 samples/s/p  25 days, 2:07:47 }
- INFO - opt_global_step: 0, skip_data_grad_norm_threshold: 3.0, is_skip: [ True]
ValueError: Current global norm [47.329006] of step 2 has been 2 consecutive times greater than threshold 3.0, stop training...
```

## 权重健康监测

### 概述

MindSpore Transformers提供的健康监测功能，能够通过监测stage0下的embedding local norm，来判定保存的权重的健康情况，通过文件health_ckpts.json，来记录训练过程中所有保存的权重的健康状况，续训时通过该文件自动寻找最新的健康的权重进行续训。

本功能涵盖以下三个步骤：

1. 打开健康监测开关，通过一段时间的正常训练来确定需要设定的embedding local norm的阈值。
2. 设定阈值后重新开启训练，当保存权重时，embedding local norm超过阈值，则记录权重健康状况为不健康，反之则记录为健康，记录中1表示不健康，0表示健康。
3. 续训时，自动根据上次训练生成的health_ckpts.josn文件中记录的最新的健康权重进行续训。

**注意**：

- 只有当pipeline stage>1时的stage0下的embedding norm才有意义。
- 只有stage0下的卡的权重才有对应的健康状况，记录文件记录的是所有卡权重汇总后的结果，即只要有一张卡的权重的健康状况为不健康，那么该步数对应的权重的健康状况则为不健康。当stage0下所有卡的权重均为健康时，文件才会记录该步数下对应的权重的健康状况为健康。
- 当记录文件中不存在健康的权重时，则会提示用户重新训练直到存在健康的权重，如若训练一直无法产生健康的权重，则应当考虑设定的embedding local norm的阈值是否合理。
- 如果指定权重进行续训，则优先以指定的权重进行续训，不考虑权重的健康状况。
- 该功能不支持full batch的场景。
- 开启该功能可能会存在通信内存不足的风险。

#### 使用方法

**注意**：以下示例所展示的参数数值仅作为实验数据，请以真实训练数据为准。

本功能通过YAML配置文件使能：

```yaml
use_checkpoint_health_monitor : True

monitor_config:
  monitor_on: True

runner_wrapper:
  local_norm: True

callbacks:
  - type: CheckpointMonitor
    save_checkpoint_steps: 1
    embedding_local_norm_threshold: 270.0

parallel:
  full_batch: False
  dataset_strategy: [[4, 1], [4, 1]]

parallel_config:
  data_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 2
```

**参数说明：**

| 参数名称                           | 描述                                                                                                                                                                                                                                                                                  | 类型    | 是否可选       | 取值范围 |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|------------|-----|
| use_checkpoint_health_monitor  | 健康监测功能开关。默认值为`False`。                                                                                                                                                                                                                                                               | bool  | 可选         |     |
| monitor_config                 | 训练指标监控配置。默认值为`None`。                                                                                                                                                                                                                                                                |       | 可选         |     |
| monitor_on                     | 是否开启训练指标监控配置，开启后才能观测embedding local norm的数据指标。默认值为`False`。                                                                                                                                                                                                                          | bool  | 可选         |     |
| runner_wrapper                 | wrapper配置。                                                                                                                                                                                                                                                                          |       | 必选         |     |
| local_norm                     | 单卡上各参数的梯度范数。默认值为`False`。                                                                                                                                                                                                                                                            | bool  | 可选         |     |
| callbacks                      | callbacks配置。                                                                                                                                                                                                                                                                        |       | 必选         |     |
| save_checkpoint_steps          | 保存权重的步数间隔。                                                                                                                                                                                                                                                                          | int   | 必选         | 正整数 |
| embedding_local_norm_threshold | 健康监测的embedding norm的阈值。默认值为`1.0`。                                                                                                                                                                                                                                                   | float | 可选         | 大于0 |
| parallel                       | 并行策略配置。                                                                                                                                                                                                                                                                             |       | 必选         |     |
| full_batch                     | 是否在并行模式下从数据集中读取加载完整的批数据，设置为`True`表示所有rank都读取完整的批数据，设置为`False`表示每个rank仅加载对应的批数据，设置为`False`时必须设置对应的`dataset_strategy`。此功能仅支持`False`。                                                                                                                                                  |    bool   | 必选 `False` |     |
| dataset_strategy               | 仅支持`List of List`类型且仅在`full_batch=False`时生效，列表中子列表的个数需要等于`train_dataset.input_columns`的长度，并且列表中的每个子列表需要和数据集返回的数据的shape保持一致。一般在数据的第1维进行数据并行切分，所以子列表的第1位数配置与`data_parallel`相同，其他位配置为`1`。具体原理可以参考[数据集切分](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dataset_slice.html)。 |   list    | 必选         |     |
| parallel_config                | 并行参数配置。                                                                                                                                                                                                                                                                             |       | 必选         |     |
| data_parallel                  | 设置数据并行数。                                                                                                                                                                                                                                                                            |   int    | 必选         | 正整数 |
| pipeline_stage                 | 设置流水线并行数。                                                                                                                                                                                                                                                                           |    int   | 必选         | 正整数 |
| micro_batch_num                | 设置流水线并行的微批次大小，在`parallel_config.pipeline_stage`大于1时，应满足`parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage`。                                                                                                                                                       |    int   | 必选         | 正整数 |

### 使用示例

假设以Llama3.1-8B为例子，使用的[finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml)按照上述[配置](#使用方法-1)添加参数和修改，其余步骤请参考[Llama3.1-8B文档](https://gitee.com/lanshaozuishuai/mindformers/blob/dev/research/llama3_1/README.md)。开启训练：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/llama3_1 \
    --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
    --train_data /{path}/wiki4096.mindrecord \
    --run_mode train \
    --use_parallel True" 8
```

模型正式开始训练时，日志会打印当前步数的embedding local norm，便于用户统计观测后设定阈值。

```log
- INFO - { Epoch:[  1/  2], step:[    1/ 6500], loss: 0.000, per_step_time: 157149ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [44.31202], train_throughput_per_npu: 3.023T
- INFO -    0.0% |                                                  | 0.00636 samples/s/p  23 days, 15:26:22 }
- INFO - embedding_local_norm: 251.79117

- INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 0.000, per_step_time: 8266ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [47.328575], train_throughput_per_npu: 57.471T
- INFO -    0.0% |                                                  | 0.12096 samples/s/p  1 day, 5:50:52 }
- INFO - embedding_local_norm: 291.3603
```

health_ckpts.json记录数据如下：

ckpt_name记录的为权重文件名，is_health记录的是对应权重的健康状况。记录中1表示不健康，0表示健康。

```json
[
    {
        "is_health": 0,
        "ckpt_name": "llama3_1_8b_rank_0-1_1.safetensors"
    },
    {
        "is_health": 1,
        "ckpt_name": "llama3_1_8b_rank_0-2_1.safetensors"
    }
]
```