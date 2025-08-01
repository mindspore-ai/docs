# Training High Availability

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/high_availability.md)

## Overview

MindSpore Transformers high availability provides the following six functions:

- **End-of-life CKPT**: It is mainly aimed at accelerating the fault recovery in the training process of large models. This feature verifies the integrity and consistency of the intermediate state data after a fault occurs during the training process and generates an end-of-life CheckPoint data, which can be used to recover the training and reduce the loss of training iterations caused by the fault.
- **UCE Fault-tolerant Recovery**: It mainly focuses on the detection of UCE faults in on-chip memory during the training process of large models, and accomplishes online repair to reach Step-level recomputation.
- **HCCE Fault-tolerant Recovery**: It mainly focuses on hccl recompute error during the training process of large models, and accomplishes online repair to reach Step-level recomputation.
- **TRE Training Result Excepition Recovery**：It mainly focuses on the detection of value excepton of loss, global-norm, etc. during the training process of large models, and accomplishes online repair to reach Step-level recomputation.
- **ARF Process-Level Rescheduling Recovery**: Instead of pulling up the entire cluster again after an anomaly in training occurs, simply restart or replace it on a node-by-node basis to complete the repair and continue training.
- **TSP Training Step Pause Function**：After each training step is completed, enter the train pause interface，pause or resume training according to the needs of upper level operations. For example, pause training to perform communication network track switching, and resume training after successful switching.

Constraints and dependencies of the high availability functions:

| | End-of-life CKPT | UCE | HCCE | ARF | TRE | TSP |
| - | - | - | - | - | - | - |
| Depending on MindIO | Yes | Yes | Yes | Yes | No | Yes |
| Replica relationship between between cards | Yes | Yes | No | Yes | No | No |
| Sink Size is 1 | Yes | Yes | Yes | Yes | No | No |

These six high availability functions are currently only supported in the MindSpore Ascend back-end graph schema to support Step-level recovery.

The replica relationship between cards is used to make sure when one of the cards fails, it can be recovered from the other card. It requires that there must be at least two copies of redundancy in both the weights and the optimizer. To ensure this redundancy relationship, data parallelism must be turned on to ensure that there are two cards with the same weights, and also if optimizer parallelism is turned on, it must be ensured that there are two cards with the same optimizer state.

When End-of-life CKPT, UCE and ARF functions are turned on in combination, the order in which they take effect is: UCE -> ARF -> End-of-Life CKPT, and if one of the functions can be recovered, the next function will not be executed. The end-of-life CKPT function serves as a final safeguard, and the entire training process exits upon completion of this function, so it will be turned on by default when the UCE or ARF functions are turned on.

The rapid recovery of faults is a combination of ARF and TRE functions, with the order of effectiveness being TRE -> ARF. TRE is responsible for monitoring outliers in the global norm and throwing them, while ARF is responsible for capturing TRE anomalies and restarting the corrective cluster for training without interrupting the entire process.

Quick recovery and use instructions for malfunctions:

> - The process-level rapid recovery feature can effectively reduce the time required to restart training after encountering abnormal global norms during the training process.
> - Please train normally for a period of time before use to determine the threshold of the global norm that needs to be set.
> - Once a global norm exceeding the set threshold is encountered, an exception will be thrown immediately, entering the fast recovery phase.
> - The data skipping function cannot be used in conjunction with the quick fault recovery function. Refer to the data skipping function in [Data Skip](https://www.mindspore.cn/mindformers/docs/en/master/feature/skip_data_and_ckpt_health_monitor.html#skipping-data) function.

## Instructions for Use

The high availability feature switch is enabled by an environment variable, and the switch is not set separately in the YAML configuration file. For high availability functions which depend on replica relationship between between cards, the YAML file needs to be able to configure the weights and optimizer states to be the same for both cards, as detailed in the [Replica Relationships Configuration](#replica-relationships-configuration) section of this document.

For high availability functions which depend on MindIO, the user needs to install the MindIO TFT SDK package. Please refer to [Install MindIO TFT SDK on compute nodes](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft011.html).

### Environment Variable Configuration

```shell
export MINDIO_FOR_MINDSPORE=1
export MS_ENABLE_TFT="{TTP:1,UCE:1,HCCE:1,ARF:1,TRE:1,TSP:1}"
export MS_TFT_IP=127.0.0.1
export MS_TFT_PORT=30051
```

- `MINDIO_FOR_MINDSPORE`: Enabling MindIO TFT SDK to support MindSpore
- `MS_ENABLE_TFT`: Indicates that the TTP, UCE, ARF, TRE and TSP functions are enabled. If you want to enable only one of these functions, set the corresponding value to 1.
    - **TTP (Try To Persist)**: End-of-life CKPT function
    - **UCE (Uncorrectable Memory Error)**: UCE fault tolerance recovery
    - **HCCE (Huawei Collective Communication Error)**: HCCL recompute error recovery
    - **ARF (Air Refuelling)**: Process-level rescheduling recovery function
    - **TRE (Training Result Error)**: Training result exception recovery
    - **TSP (Training Step Pause)**：Training step pause function
    - When UCE or ARF is enabled, TTP is enabled by default.
    - Enabling both TRE and asynchronous CKPT features at the same time cannot guarantee that the loss before and after resuming training is exactly the same.
    - TRE does not depend on MindIO. It is not necessary to configure the MindIO-related environment variables MINDIO_FOR_MINDSPORE, MS_TFT_IP, and MS_TFT_PORT to enable only the TRE feature

- `MS_TFT_IP` and `MS_TFT_PORT` represent the IP and port number of TFT Controller respectively, no default value, need to be specified by user. If the Controller is started by MindSpore Transformers, the IP and port number of the rank0 node in the user's cluster are configured. If the Controller is started by the user, configure the IP and port number of the Controller.

### YAML Configuration

The YAML configuration consists of two parts: the end-of-life CKPT saving and recovery configuration and the replica relationship between cards configuration.

#### Saving and Restoring Configurations

The end-of-life CheckPoint preservation and recovery capabilities are used for initial and renewal training respectively, which reuse the existing MindSpore Transformers configuration, and the following describes the configuration for initial and renewal training respectively.

- **Initial Training Configuration**

    ```yaml
    output_dir: './output' # The directory where CheckPoints and Strategies are stored
    load_checkpoint: ''    # Configuration is empty for initial training
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: False  # Configuration is False for initial training
    run_mode: 'train'

    callbacks:
      - type: CheckpointMonitor
        prefix: "llama2_13b"
        save_checkpoint_steps: 100
        integrated_save: False
        async_save: False
    ```

- **Renewal Training Configuration**

    ```yaml
    output_dir: './output' # The directory where CheckPoints and Strategies are stored
    load_checkpoint: './output/checkpoint/'   # Configure CheckPoint paths during renewal training
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: True  # Configured to True for renewal training
    run_mode: 'train'

    callbacks:
      - type: CheckpointMonitor
        prefix: "llama2_13b"
        save_checkpoint_steps: 100
        integrated_save: False
        async_save: False
    ```

#### Replica Relationships Configuration

The key to the end-of-life CheckPoint, UCE and ARF functions of high availability is to configure the weight and optimizer copy redundancy relationship. The core of the configuration is that the dimension of the data parallel domain is greater than 2, and if you overlay the optimizer parallelism, you need to ensure that the number of copies of the optimizer is greater than 2 at the same time. So the configuration is divided into two categories, with the optimizer parallelism and without the optimizer parallelism. The following is an example of how to configure 8 cards.

- **Without the Optimizer Parallelism**

    Data parallelism dp configured as a multiple of 2 is sufficient, so that there will exist two cards with the same weights and optimizer state.

    ```yaml
    parallel:
      enable_parallel_optimizer: False
    parallel_config:
      data_parallel: 2
      model_parallel: 4
      pipeline_stage: 1
    ```

- **With the Optimizer Parallelism**

    After turning on the optimizer parallelism you must ensure that a copy of the optimizer state exists, the key to configure is optimizer_weight_shard_size to 2. The number of copies of the optimizer state at this point is data_parallel/optimizer_weight_shard_size. Therefore, if the data parallelism is configured to 2, there is no optimizer replica, and the data parallelism must be configured to 4; the number of replicas in this case is data_parallel/optimizer_weight_shard_size = 4/2 = 2.

    ```yaml
    parallel:
      enable_parallel_optimizer: True
      parallel_optimizer_config:
        optimizer_weight_shard_size: 2
    parallel_config:
      data_parallel: 4
      model_parallel: 2
      pipeline_stage: 1
    ```

## Example Usage

### End-of-life CheckPoint

This section demonstrates the use of the end-of-life CKPT using Llama2-13B training as an example.

1. First install MindSpore and MindIO
2. Download MindSpore Transformers and modify the `configs/llama2/pretrain_llama2_13b_bf16.yaml` configuration file with the following main configuration:

    ```yaml
    # runner config
    runner_config:
      epochs: 2
      batch_size: 4
      sink_mode: True
      sink_size: 1

    # ......

    # parallel context config
    parallel:
      parallel_mode: 1 # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
      gradients_mean: False
      enable_alltoall: False
      full_batch: True
      search_mode: "sharding_propagation"
      enable_parallel_optimizer: True
      strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
      parallel_optimizer_config:
        gradient_accumulation_shard: False
        parallel_optimizer_threshold: 64
        optimizer_weight_shard_size: 4

    # ......

    # default parallel of device num = 16 for Atlas 800T A2
    parallel_config:
      data_parallel: 8
      model_parallel: 1
      pipeline_stage: 1
      use_seq_parallel: False
      micro_batch_num: 1
      vocab_emb_dp: True
      gradient_aggregation_group: 4
    ```

    The following key points need to be noted:

    - `sink_size: 1`: Features such as end-of-life CKPT and UCE fault-tolerant recovery do not support scenarios where `sink_size` is greater than 1, so it is configured as 1 here.
    - `enable_parallel_optimizer: True`: Enable optimizer parallelism.
    - `optimizer_weight_shard_size: 4`: The slice size of optimizer parallelism is 4.
    - `data_parallel: 8`: Data parallelism is configured as 8.

    As explained in the previous section, the value of `data_parallel/optimizer_weight_shard_size` is `8 / 4 = 2`, which is greater than 1, so there is a replica relationship.
3. Execute the following command to start the training

    ```bash
    export MINDIO_FOR_MINDSPORE=1

    export MS_ENABLE_TFT="{TTP:1,UCE:1,ARF:1,TSP:1}"
    export MS_TFT_IP=127.0.0.1
    export MS_TFT_PORT=30051

    bash scripts/msrun_launcher.sh "run_mindformer.py \
      --config configs/llama2/pretrain_llama2_13b_bf16.yaml \
      --train_dataset_dir "/YourDataSetPath" \
      --use_parallel True --run_mode train" 8
    ```

    Note: You need to replace `/YourDataSetPath` with the path of the actual dataset.
4. After a few steps of training, terminate the worker process and trigger an end-of-life CKPT save

    Note: With the above startup method, the MindIO Controller is attached to worker 0. In this case, worker 0 cannot be terminated, or else the MindIO Controller will exit and the end-of-life CKPT cannot be triggered. However, when training is started via taskd, the MindIO Controller is a separate process and the worker 0 process can be terminated.
5. Confirm end-of-life CheckPoint generation

    At the end of the entire training process, the reasonableness of the final generated CheckPoint file is confirmed through the log as follows:

    1). Execute the command `find output/checkpoint/ -name '*.ckpt'` to find the generated CheckPoint file:

    ```text
    $ find output/checkpoint/ -name '*.ckpt'
    output/checkpoint/rank_2/llama2_13b_rank_2-5_1.ckpt
    output/checkpoint/rank_3/llama2_13b_rank_3-5_1.ckpt
    output/checkpoint/rank_0/llama2_13b_rank_0-5_1.ckpt
    output/checkpoint/rank_5/llama2_13b_rank_5-5_1.ckpt
    ```

    2). Execute the command `cat output/msrun_log/worker_0.log | grep 'Epoch:'` to see the trained steps:

    ```text
    $ cat output/msrun_log/worker_0.log | grep 'Epoch:'
    2025-04-07 15:34:27,308 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    1/   19], loss: 10.649, per_step_time: 103328ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [1 31049], train_throughput_per_npu: 2.896T
    2025-04-07 15:34:29,173 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    2/   19], loss: 10.633, per_step_time: 1752ms, lr: 1e-05, overflow cond: False, loss_scale: 1.0, global_norm: [1 508834], train_throughput_per_npu: 170.738T
    2025-04-07 15:34:30,941 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    3/   19], loss: 9.673, per_step_time: 1754ms, lr: 9.981987e-06, overflow cond: False, loss_scale: 1.0, global_norm [10.579812], train_throughput_per_npu: 170.523T
    2025-04-07 15:34:32,704 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    4/   19], loss: 9.287, per_step_time: 1756ms, lr: 9.928079e-06, overflow cond: False, loss_scale: 1.0, global_norm [21.932272], train_throughput_per_npu: 170.319T
    2025-04-07 15:34:34,469 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    5/   19], loss: 8.867, per_step_time: 1758ms, lr: 9.8386645e-06, overflow cond: False, loss_scale: 1.0, global_norm [16.986555], train_throughput_per_npu: 170.173T
    ```

    3). Execute the command `cat output/msrun_log/worker_0.log | grep 'report group list:'` to see the replica relationships of MindIO output in the log:

    ```text
    $ cat output/msrun_log/worker_0.log | grep 'report group list:'
    2025-04-07 15:34:27.363613 info 1879138 [TTP controller.cpp:1512] rank:4, report group list: [0, 4]
    2025-04-07 15:34:27.385564 info 1879139 [TTP controller.cpp:1512] rank:7, report group list: [3, 7]
    2025-04-07 15:34:27.393198 info 1879136 [TTP controller.cpp:1512] rank:6, report group list: [2, 6]
    2025-04-07 15:34:27.393515 info 1879142 [TTP controller.cpp:1512] rank:1, report group list: [1, 5]
    ```

    From the training step information above, we can see that the 5 steps that have been trained, and the number is the same as the 5 in the file name `llama2_13b_rank_2-5_1.ckpt` of CheckPoint.

    The copy relations `[0, 4]`, `[3, 7]`, `[2, 6]` and `[1, 5]` are known from the output in the log:

    - The rank 0 and rank 4 weights have a replica relationship, and the end-of-life checkpoint is stored in rank 0.
    - The rank 3 and rank 7 weights have a replica relationship, and the end-of-life checkpoint is stored in rank 3.
    - The rank 2 and rank 6 weights have a replica relationship, and the end-of-life checkpoint is stored in rank 2.
    - There is a replica relationship between rank 1 and rank 5 weights, and since worker 1 terminates, the final checkpoint is stored in rank 5.

### Abnormal Training Results Recovery

This chapter uses Llama3.1-8B training as an example to demonstrate the use of rapid fault recovery.

> The parameter values shown in the following examples are only experimental data, please refer to real training data.

1. Install [MindSpore](https://www.mindspore.cn/install/en) first.
2. Download MindSpore Transformers, using [finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/master/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml) to add and modify parameters according to the configuration below:

   ```yaml
    output_dir: './output'

    monitor_config:
      monitor_on: True
      check_for_global_norm: True
      global_norm_spike_threshold: 44.0

    callbacks:
      - type: CheckpointMonitor
        save_checkpoint_steps: 1
    ```

    **Parameter：**

    | Parameters                  | Description                                                                                                                                           | Type  | Optional        |
    |-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------|-----------------|
    | output_dir                  | Path to save checkpoint/strategy. Default to `./output`.                                                                                              | str   | Optional        |
    | monitor_config              | Whether to enable training indicator monitoring configuration. Default to `None`.                                                                     | dict  | Optional        |
    | monitor_on                  | Whether to enable training metric monitoring configuration. Only when enabled can abnormal global norm be monitored and TRE functionality be enabled. | bool  | Required `True` |
    | check_for_global_norm       | Whether to enable the process-level fault rapid recovery function is mutually exclusive with the data skip function. Default to `False`.              | bool  | Optional        |
    | global_norm_spike_threshold | The threshold for global norm, which triggers data skipping when global norm is exceeded. Default to `3.0`.                                           | float | Optional        |
    | callbacks                   | The configs of callbacks.                                                                                                                             | list  | Required        |
    | save_checkpoint_steps       | The step interval for saving weights.                                                                                                                 | int   | Required        |

3. Configure environment variables:

   ```shell
   export MS_ENABLE_TFT="TRE:1"
   ```

4. Run the following command to start training:

    ```shell
    cd mindformers

    bash scripts/msrun_launcher.sh "run_mindformer.py \
        --register_path research/llama3_1 \
        --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
        --train_data /{path}/wiki4096.mindrecord \
        --run_mode train \
        --use_parallel True" 8
    ```

5. When the model officially starts training and encounters a global norm greater than the set threshold, the following log will be printed to prompt the user that an abnormal global norm has been encountered, and the corresponding global step and global norm will be recorded in abnormal_global_norm.json, triggering an error and entering the fast recovery phase.

    ```text
    - INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 11.905, per_step_time: 2775ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [45.702465], train_throughput_per_npu: 171.176T
    - INFO -    0.0% |                                                  | 0.36029 samples/s/p  10:01:16 }
    - INFO - Current global norm [45.702465] is greater equal than threshold 44.0, stop training...
    ```

6. After retraining, the training will continue from the previous breakpoint step count. If the global norm is still greater than the set threshold, since the corresponding global step has already been recorded in the abnormal_global_norm.json under the output dir set by YAML, only the corresponding global norm will be recorded here and it will not raise error.

    ```text
    - INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 11.905, per_step_time: 3504ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [45.706497], train_throughput_per_npu: 135.552T
    - INFO -    0.0% |                                                  | 0.28531 samples/s/p  12:39:17 }
    - INFO - The global norm [45.706497] of step 2 is still greater or equal than threshold 44.0, continue training.
    ```

    The data recorded in abnormal_global_norm.json is as follows:

    ```json
    {
      "2": [45.70246505737305, 45.70649719238281]
    }
    ```

    '2' represents the global step corresponding to the number of training steps, and the following list records the global norm of training before and after recovery.