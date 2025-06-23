# Data Skip And Checkpoint Health Monitor

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/skip_data_and_ckpt_health_monitor.md)

## Overview

The data skipping function refers to the process where, during the training process, when the parameter global norm exceeds the set threshold, it accumulates the number of out of bounds and skips the training data for the current step, and proceeds to retraining in the next step; When the cumulative number of violations reaches the threshold, an abnormal interrupt will be triggered to terminate the training. The health monitoring function refers to monitoring the health status of the saved weights when saving them, generating a file to record the health status of the weights, and using this file to select the latest healthy weights for the next training session.

Please refer to [Checkpoint Health Monitor](#checkpoint-health-monitor) for the determination of weight health status.

> - The combination of data skipping function and health monitoring function can effectively solve the problem of data anomalies caused by abnormal global norm during the training process. Before use, please train normally for a period of time to determine the threshold of the global norm that needs to be set, the threshold of the number of consecutive anomalies, and the threshold of the embedding norm.
> - Please note that training will only be interrupted when there are consecutive exceptions. If there is only one instance where it returns to normal, the cumulative count will be cleared. Therefore, please control the threshold setting.
> - The data skipping function cannot be used in conjunction with the quick fault recovery function. Refer to the process level rescheduling recovery function in the [high availability feature](https://www.mindspore.cn/mindformers/docs/en/dev/feature/high_availability.html).

## Skipping Data

### Overview

MindSpore Transformers provides the function of skipping data, which can skip the current training data when there is a global norm exception, and trigger an exception interrupt when the number of consecutive exceptions reaches the set threshold.

This feature has the following three behaviors in total:

- An out of bounds global norm has occurred, with a cumulative abnormal occurrence of +1. Skipping the current step training data and printing log information.
- global norm has returned to normal, and the cumulative number of abnormal occurrences has been cleared.
- When the cumulative number of abnormal occurrences reaches the set threshold, an abnormal interrupt is triggered and the training is terminated.

#### Usage

**Note**: The parameter values shown in the following examples are only experimental data, please refer to real training data.

This feature is enabled through YAML configuration files:

```yaml
use_skip_data_by_global_norm: True

monitor_config:
  monitor_on: True
  check_for_global_norm: False
  global_norm_spike_threshold: 3.0
  global_norm_spike_count_threshold: 10
```

**Parameter:**

| Parameter                         | Description                                                                                                                                                                | Type  | Optional | Value Range      |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|---------|------------------|
| use_skip_data_by_global_norm      | Data skip function switch. Default to `False`.                                                                                                                             | Bool  | Optional     |                  |
| monitor_config                    | Training indicator monitoring configuration. Default to `None`.                                                                                                            |       | Optional     |                  |
| monitor_on                        | Whether to enable training metric monitoring configuration. Default to `False`.                                                                                            | Bool  | Optional     |                  |
| check_for_global_norm             | To enable the fault recovery function, which is mutually exclusive with the data skipping function. Default to `False`.                                                    | Bool  | Optional     |                  |
| global_norm_spike_threshold       | The threshold for global norm, which triggers data skipping when global norm is exceeded. Default to `3.0`.                                                                | Float | Optional     | Greater than 0   |
| global_norm_spike_count_threshold | The number of consecutive abnormal global_norm. When the number reaches the threshold, an abnormal interruption is triggered, and training is terminated. Default to `10`. | Int   | Optional     | Positive integer |

### Conversion Example

Assuming Llama3.1-8B is taken as an example, use [finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml) to add parameters according to the above [Configuration](#usage), please refer to the [Llama3.1-8B Document](https://gitee.com/lanshaozuishuai/mindformers/blob/dev/research/llama3_1/README.md) for the remaining steps. Start training:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/llama3_1 \
    --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
    --train_data /{path}/wiki4096.mindrecord \
    --run_mode train \
    --use_parallel True" 8
```

When the model officially starts training, if the global norm is greater than the set threshold, the following log will be printed, indicating that the user has experienced abnormal global norm n times in a row and skipped the training data for the current step count.

```log
- INFO - { Epoch:[  1/  2], step:[    1/ 6500], loss: 0.000, per_step_time: 166756ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [44.313248], train_throughput_per_npu: 2.849T
- INFO -    0.0% |                                                  | 0.00600 samples/s/p  25 days, 2:07:47 }
- INFO - opt_global_step: 0, skip_data_grad_norm_threshold: 3.0, is_skip: [ True]
- INFO - Current global norm [44.313248] of step 1 has been 1 consecutive times greater than threshold: 3.0
```

When the number of consecutive exceptions reaches the set threshold, print an error log and terminate the training.

```log
- INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 0.000, per_step_time: 7637ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [47.329006], train_throughput_per_npu: 62.211T
- INFO -    0.0% |                                                  | 0.00600 samples/s/p  25 days, 2:07:47 }
- INFO - opt_global_step: 0, skip_data_grad_norm_threshold: 3.0, is_skip: [ True]
ValueError: Current global norm [47.329006] of step 2 has been 2 consecutive times greater than threshold 3.0, stop training...
```

## Checkpoint Health Monitor

### Overview

The health monitoring function provided by MindSpore Transformers can determine the health status of saved weights by monitoring the embeddings in stage0. The health status of all saved weights during the training process is recorded in the file health_ckpts.json, and the latest healthy weights are automatically found through this file for further training.

This feature covers the following three steps:

1. Turn on the health monitoring switch and determine the threshold for the embeddings needed to be set through a period of normal training.
2. After setting the threshold, restart the training. When the embeddings exceed the threshold when saving weights, the health status of the weights is recorded as unhealthy. Otherwise, it is recorded as healthy, with 1 indicating unhealthy and 0 indicating healthy.
3. When resuming training, the latest health weights recorded in the health_ckpts.json file generated from the previous training will be automatically used for continuation.

**Note**:

- Only the embedding norm under stage0 is meaningful when pipeline stage is greater than 1.
- Only the weights of cards in stage 0 have corresponding health status. The record file records the total health status of all card weights, that is, if the health status of a card's weight is unhealthy, then the health status of the weight corresponding to that step is unhealthy. Only when the weights of all cards in stage 0 are healthy, will the file record the health status of the corresponding weights for that step as healthy.
- When there are no health weights in the record file, the user will be prompted to retrain until there are health weights. If the training fails to generate health weights, the threshold set for embeddings should be considered whether it is reasonable.
- If a weight is specified for resuming training, priority will be given to the specified weight for resuming training, without considering the health status of the weight.
- This feature does not support full batch scenarios.
- Enabling this feature may pose a risk of insufficient communication memory.

#### Usage

**Note**: The parameter values shown in the following examples are only experimental data, please refer to real training data.

This feature is enabled through YAML configuration files:

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

**Parameter:**

| Parameter                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Type | Optional         | Value Range      |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------|------------------|
| use_checkpoint_health_monitor  | Checkpoint health monitoring function switch. Default to `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Bool | Optional         |                  |
| monitor_config                 | Training indicator monitoring configuration. Default to `None`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |      | Optional         |                  |
| monitor_on                     | Whether to enable the training metric monitoring configuration. Only after enabling it can you observe the data metrics of embedding local norm. Default to `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Bool | Optional         |                  |
| runner_wrapper                 | The configs of wrapper.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |      | Required         |                  |
| local_norm                     | The gradient norm of each parameter on a single card. Default to `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Bool | Optional         |                  |
| callbacks                      | The configs of callbacks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |      | Required         |                  |
| save_checkpoint_steps          | The step interval for saving weights.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Int  | Required         | Positive Integer |
| embedding_local_norm_threshold | The threshold of embedding norm for health monitoring. Default to `1.0`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Float | Optional         | Greater than 0   |
| parallel                       | Parallel strategy configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |      | Required         |                  |
| full_batch                     | Whether to load the full batch of data from the dataset in parallel mode. Setting it to `True` means all ranks will load the full batch of data. Setting it to `False` means each rank will only load the corresponding batch of data. When set to `False`, the corresponding `dataset_strategy` must be configured. This feature only supports`False`.                                                                                                                                                                                                                                                                | Bool | Required `False` |                  |
| dataset_strategy               | Only supports `List of List` type and is effective only when `full_batch=False`. The number of sublists in the list must be equal to the length of `train_dataset.input_columns`. Each sublist in the list must have the same shape as the data returned by the dataset. Generally, data parallel splitting is done along the first dimension, so the first dimension of the sublist should be configured to match `data_parallel`, while the other dimensions should be set to `1`. For detailed explanation, refer to [Dataset Splitting](https://www.mindspore.cn/tutorials/en/master/parallel/dataset_slice.html). | List | Required               |                  |
| parallel_config                | Parallel parameter configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |      | Required               |                  |
| data_parallel                  | Set the number of data parallel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Int  | Required               | Positive Integer              |
| pipeline_stage                 | Set the number of pipeline parallel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Int  | Required               | Positive Integer              |
| micro_batch_num                | Set the pipeline parallel microbatch size, which should satisfy `parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage` when `parallel_config.pipeline_stage` is greater than 1.                                                                                                                                                                                                                                                                                                                                                                                                                         | Int  | Required               | Positive Integer              |

### Conversion Example

Assuming Llama3.1-8B is taken as an example, use [finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml) to add parameters and modify according to the above [Configuration](#usage-1), please refer to the [Llama3.1-8B Document](https://gitee.com/lanshaozuishuai/mindformers/blob/dev/research/llama3_1/README.md) for the remaining steps. Start training:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/llama3_1 \
    --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
    --train_data /{path}/wiki4096.mindrecord \
    --run_mode train \
    --use_parallel True" 8
```

When the model officially starts training, the log will print the embedding local norm for the current number of steps, making it easier for users to set thresholds after statistical observation.

```log
- INFO - { Epoch:[  1/  2], step:[    1/ 6500], loss: 0.000, per_step_time: 157149ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [44.31202], train_throughput_per_npu: 3.023T
- INFO -    0.0% |                                                  | 0.00636 samples/s/p  23 days, 15:26:22 }
- INFO - embedding_local_norm: 251.79117

- INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 0.000, per_step_time: 8266ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [47.328575], train_throughput_per_npu: 57.471T
- INFO -    0.0% |                                                  | 0.12096 samples/s/p  1 day, 5:50:52 }
- INFO - embedding_local_norm: 291.3603
```

The recorded data of health_ckpts.json is as follows:

The ckpt_name records the weight file name, while is_health records the health status of the corresponding weight. In the record, 1 represents unhealthy and 0 represents healthy.

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