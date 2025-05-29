# High Availability

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/high_availability.md)

## Overview

MindSpore Transformers high availability provides the following three functions:

- **End-of-life CKPT**: It is mainly aimed at accelerating the fault recovery in the training process of large models. This feature verifies the integrity and consistency of the intermediate state data after a fault occurs during the training process and generates an end-of-life CheckPoint data, which can be used to recover the training and reduce the loss of training iterations caused by the fault.
- **UCE Fault-tolerant Recovery**: It mainly focuses on the detection of UCE faults in on-chip memory during the training process of large models, and accomplishes online repair to reach Step-level recomputation.
- **Process-Level Rescheduling Recovery**: Instead of pulling up the entire cluster again after an anomaly in training occurs, simply restart or replace it on a node-by-node basis to complete the repair and continue training.

The high availability feature is currently only supported in the MindSpore Ascend back-end graph schema; this feature also needs to support Step-level recovery, so only a sink_size of 1 is supported when configuring data sinking.

The high availability feature is based on the existence of a replica relationship between the two cards so that when one of the cards fails, it can be recovered from the other card, and therefore there will be two copies of redundancy in both the weights and the optimizer, which will take up more video memory. To ensure this redundancy relationship, data parallelism must be turned on to ensure that there are two cards with the same weights, and also if optimizer parallelism is turned on, it must be ensured that there are two cards with the same optimizer state.

All three functions can be turned on at the same time or individually. When these three functions are turned on in combination, the order in which they take effect is: UCE Fault Tolerance Recovery -> Process-Level Rescheduling Recovery -> End-of-Life CKPT, and if one of the functions can be recovered, the next function will not be executed. The end-of-life CKPT function serves as a final safeguard, and the entire training process exits upon completion of this function, so it will be turned on by default when the other two functions are turned on.

The end-of-life CKPT saving of the Checkpoint file and the renewal of training from that file use the existing MindSpore Transformers capabilities in the same way, except that the end-of-life CKPT relies on the strategy file, so that folder needs to be configured for both the training and the renewal of the training.

When an exception triggers an end-of-life CheckPoint save, if de-redundant saving is not turned on, only one card in each data parallel field saves the CheckPoint, and the rest of the cards do not save the CheckPoint. Therefore, when resuming training, it is also necessary to enable the high availability feature in order to resume, otherwise the other cards will not be able to find the available CheckPoint and will report an error exit. Users can determine whether a CheckPoint is triggered by the end-of-life CKPT feature by calculating whether the number of CheckPoints saved by the distribution is less than the number of clusters.

## Instructions for Use

The high availability feature switch is enabled by an environment variable, and the switch is not set separately in the YAML configuration file, but the YAML file needs to be able to configure the weights and optimizer states to be the same for both cards, as detailed in the [Replica Relationships Configuration](#replica-relationships-configuration) section of this document.

The high availability feature relies on the user to install the MindIO TFT SDK package. Please refer to [Install MindIO TFT SDK on compute nodes](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft011.html).

### Environment Variable Configuration

```shell
export MINDIO_FOR_MINDSPORE=1
export MS_ENABLE_TFT="{TTP:1,UCE:1,ARF:1}"
export MS_TFT_IP=127.0.0.1
export MS_TFT_PORT=30051
```

- `MINDIO_FOR_MINDSPORE`: Enabling MindIO TFT SDK to support MindSpore
- `MS_ENABLE_TFT`: Indicates that the TTP, UCE and ARF functions are enabled. If you want to enable only one of these functions, set the corresponding value to 1.
    - **TTP (Try To Persist)**: End-of-life CKPT function
    - **UCE (Uncorrectable Memory Error)**: UCE fault tolerance recovery
    - **ARF (Air Refuelling)**: Process-level rescheduling recovery function
    - When UCE or ARF is enabled, TTP is enabled by default.

- `MS_TFT_IP` and `MS_TFT_PORT` represent the IP and port number of TFT Controller respectively, no default value, need to be specified by user. If the Controller is started by MindSpore Transformers, the IP and port number of the rank0 node in the user's cluster are configured. If the Controller is started by the user, configure the IP and port number of the Controller.

### YAML Configuration

The YAML configuration consists of two parts: the end-of-life CKPT saving and recovery configuration and the highly available replica relationship configuration.

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

The key to the three functions of high availability is to configure the weight and optimizer copy redundancy relationship. The core of the configuration is that the dimension of the data parallel domain is greater than 2, and if you overlay the optimizer parallelism, you need to ensure that the number of copies of the optimizer is greater than 2 at the same time. So the configuration is divided into two categories, with the optimizer parallelism and without the optimizer parallelism. The following is an example of how to configure 8 cards.

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

#### Examples

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

    export MS_ENABLE_TFT="{TTP:1,UCE:1,ARF:1}"
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