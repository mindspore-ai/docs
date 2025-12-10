# 训练高可用

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/mindformers/docs/source_zh_cn/feature/high_availability.md)

## 概述

MindSpore Transformers 高可用特性提供了如下几个功能：

- **临终 CKPT 功能**：主要针对大模型训练过程中的故障恢复加速。该特性在训练过程中发生故障后，校验中间状态数据的完整性和一致性，生成一次临终 Checkpoint 数据。恢复训练时能够通过该 Checkpoint 数据恢复，减少故障造成的训练迭代损失。
- **UCE 故障容错恢复功能**：主要针对大模型训练过程中片上内存的 UCE 故障检测，并完成在线修复，达到 Step 级重计算。
- **HCCE 故障恢复功能**：主要针对大模型训练过程中HCCL通信算子重计算失败，并完成在线修复，达到 Step 级重计算。
- **TRE 训练结果异常恢复功能**：主要针对大模型训练过程中出现loss或global norm等值异常检测，并完成在线修复，达到 Step 级重计算。
- **ARF 进程级重调度恢复功能**：训练发生异常后，不需要重新拉起整个集群，只需以节点为单位进行重启或替换，完成修复并继续训练。
- **TSP 训练迭代暂停功能**：在每个训练step结束后，进入训练暂停接口，根据上层运维需要进行训练暂停和继续。例如，暂停训练执行通信网络轨道切换，切换成功后继续训练。
- **RSC POD 级重调度功能**：主要作为其他快恢特性执行失败之后的兜底方案，kill故障进程以及其他正常进程（正常进程所在pod不会被kill），将故障pod从当前集群中隔离，同时调度新的pod加入集群，并恢复训练（当前版本必须依赖MindX）。

这几个高可用特性的**约束**和**依赖**如下：

| | 临终 CKPT | UCE | HCCE | ARF | TRE | TSP | RSC |
| - | - | - | - | - | - | - | - |
| 依赖MindIO组件 | Yes | Yes | Yes | Yes | No | Yes | No |
| 卡间存在副本关系 | Yes | Yes | No | Yes | No | No | No |
| Sink Size 为 1 | Yes | Yes | Yes | Yes | No | No | No |

目前这几个高可用特性只支持Ascend后端上图模式的Step级别恢复。

卡间存在副本关系的目的是当其中一张卡发生故障时，可从另外一张卡恢复。要求权重和优化器状态都会存在至少两份冗余。为保证这种冗余关系，必须开启数据并行，保证有两张卡权重一致。同时，如果开启了优化器并行，也必须确保存在两张卡的优化器状态一致。

临终 CKPT、UCE 和 ARF 组合开启这三个功能时，依次生效的顺序是：UCE -> ARF -> 临终 CKPT。如果其中一个功能可以恢复，就不会执行下一个功能。临终 CKPT 功能作为最后的保障，完成该功能后整个训练进程会退出。所以在 UCE 或 ARF 功能开启时，会默认开启临终 CKPT。

故障快速恢复由ARF和TRE两个功能组合，生效顺序为：TRE -> ARF 。TRE负责监测global norm的异常值并抛出异常，ARF负责捕获TRE异常后重新拉起整个集群修复训练，整个过程不中断训练。

故障快速恢复使用须知：

> - 进程级快速恢复功能，能有效减少训练过程中遇到异常 global norm 而导致中断训练直至重新拉起的时间。
> - 使用前请先正常训练一段时间，从而确定需要设定的 global norm 的阈值。
> - 一旦遇到超过设定阈值的global norm，便会立即抛出异常，进入快速恢复阶段。
> - 数据跳过功能不能与故障快速恢复功能同时使用。参考[数据跳过](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/feature/skip_data_and_ckpt_health_monitor.html#数据跳过)功能。

## 使用说明

高可用特性开关由环境变量使能，YAML 配置文件中不单独设置开关。但对于要求卡间存在副本关系的高可用特性，YAML 文件需要能配置出两张卡的权重和优化器状态一致，详见本文档中的[副本关系配置](#副本关系配置)章节。

依赖MindIO组件的高可用特性需用户安装 MindIO TFT SDK 包，详细请参考[在计算节点安装 MindIO TFT SDK](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft011.html)。

### 环境变量配置

```shell
export MINDIO_FOR_MINDSPORE=1
export MS_ENABLE_TFT="{TTP:1,UCE:1,HCCE:1,ARF:1,TRE:1,TSP:1}"
export MS_TFT_IP=127.0.0.1
export MS_TFT_PORT=30051
```

- `MINDIO_FOR_MINDSPORE`：使能 MindIO TFT SDK 支持 MindSpore
- `MS_ENABLE_TFT`：表示启用训练故障容错（Training Fault Tolerance）功能。如果只想启用其中的某一个功能，则将对应的值设置为 1 即可。
    - **TTP (Try To Persist)**：临终 CKPT 功能
    - **UCE (Uncorrectable Memory Error)**：UCE 故障容错恢复功能
    - **HCCE (Huawei Collective Communication Error)**：HCCL 重计算失败恢复功能
    - **ARF (Air Refuelling)**：进程级重调度恢复功能
    - **TRE (Training Result Error)**：TRE 训练结果异常恢复功能
    - **TSP (Training Step Pause)**：TSP 训练迭代暂停功能
    - **RSC (Register Stop/Start Controller)**：POD级重调度功能
    - POD级重调度只把训练进程交给第三方组件（如 MindX）管控，仅开启RSC（当前版本必须依赖MindX）时，其他训练故障容错功能不生效
    - 开启 UCE 或者 ARF 功能时，默认开启 TTP 功能
    - 同时开启 TRE 和异步 CKPT 特性，无法保证续训前后的 loss 完全一致
    - TRE 功能不依赖 MindIO 组件，若只使能TRE特性，无需配置 MindIO 相关的环境变量 MINDIO_FOR_MINDSPORE、MS_TFT_IP 和 MS_TFT_PORT

- `MS_TFT_IP` 和 `MS_TFT_PORT` 分别表示 TFT Controller 的 IP 和端口号，无默认值，需要用户指定。如果由 MindSpore Transformers 启动 Controller，则配置用户集群中 rank0 节点的 IP 和端口号。如果用户自行启动 Controller，则配置 Controller 的 IP 和端口号。

### YAML 配置

YAML配置包含两部分：临终 CKPT 的保存及恢复配置和卡间副本关系配置。

#### 保存及恢复配置

临终的 Checkpoint 保存和恢复能力分别用于初始训练和续训，这部分复用现有的 MindSpore Transformers 的配置。以下分别介绍初始训练和续训的配置。

- **初始训练配置**

    ```yaml
    output_dir: './output' # 保存 Checkpoint 和 Strategy 的目录
    load_checkpoint: ''    # 初次训练时配置为空
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: False  # 初次训练时配置为 False
    run_mode: 'train'

    callbacks:
      - type: CheckpointMonitor
        prefix: "llama2_13b"
        save_checkpoint_steps: 100
        integrated_save: False
        async_save: False
    ```

- **续训配置**

    ```yaml
    output_dir: './output' # 保存 Checkpoint 和 Strategy 的目录
    load_checkpoint: './output/checkpoint/'   # 续训时配置 Checkpoint 路径
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: True  # 续训时配置为 True
    run_mode: 'train'

    callbacks:
      - type: CheckpointMonitor
        prefix: "llama2_13b"
        save_checkpoint_steps: 100
        integrated_save: False
        async_save: False
    ```

#### 副本关系配置

高可用的临终 CKPT、UCE 和 ARF 这三个功能的关键是配置出权重和优化器的副本冗余关系。配置的核心是数据并行域的维度大于 2，如果叠加优化器并行，需要同时保证优化器的副本数大于 2。所以配置分两类：开启优化器并行和不开启优化器并行。下面以 8 卡为例，介绍如何配置。

- **不开启优化器并行**

    数据并行度 dp 配置为 2 的倍数即可，这样就会存在两张卡的权重和优化器状态一致。

    ```yaml
    parallel:
      enable_parallel_optimizer: False
    parallel_config:
      data_parallel: 2
      model_parallel: 4
      pipeline_stage: 1
    ```

- **开启优化器并行**

    开启优化器并行后必须要保证优化器的状态存在副本，配置的关键是 optimizer_weight_shard_size 为 2。此时优化器状态的副本数为 data_parallel/optimizer_weight_shard_size。因此，如果数据并行度配置为 2 时，是不存在优化器副本的，必须把数据并行度配置为 4。此时的副本数为 data_parallel/optimizer_weight_shard_size = 4/2 = 2。

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

## 使用示例

### 临终 CKPT

本章节以 Llama2-13B 训练为例演示临终 CKPT 的使用。

1. 先安装 MindSpore 和 MindIO
2. 下载 MindSpore Transformers，修改 `configs/llama2/pretrain_llama2_13b_bf16.yaml` 配置文件，主要配置如下：

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

    需要注意以下关键点：

    - `sink_size: 1`：临终 CKPT 和 UCE 故障容错恢复等特性不支持 `sink_size` 大于 1 的场景，因此这里配置为 1。
    - `enable_parallel_optimizer: True`：使能优化器并行。
    - `optimizer_weight_shard_size: 4`：优化器并行的切分大小为 4。
    - `data_parallel: 8`：数据并行配置为 8。

    按照前面章节的说明，`data_parallel/optimizer_weight_shard_size` 的值为 `8 / 4 = 2`，大于 1，因此存在副本关系。
3. 执行下面命令启动训练

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

    注意：需要将 `/YourDataSetPath` 换成实际数据集的路径。
4. 待训练执行若干个 step 之后，终止 worker 进程，触发临终 CKPT 保存

    注意：通过上述启动方式，MindIO Controller 附着在 worker 0 进程上，此种情况下不能终止 worker 0，否则导致 MindIO Controller 退出，无法触发临终 CKPT。但是通过 taskd 方式启动训练时，MindIO Controller 是个单独的进程，可以终止 worker 0 进程。
5. 确认临终的 Checkpoint 生成

    在整个训练进程结束后，通过日志确认最终生成的 Checkpoint 文件的合理性，具体操作如下：

    1). 执行命令 `find output/checkpoint/ -name '*.ckpt'` 查找生成的 Checkpoint 文件：

    ```text
    $ find output/checkpoint/ -name '*.ckpt'
    output/checkpoint/rank_2/llama2_13b_rank_2-5_1.ckpt
    output/checkpoint/rank_3/llama2_13b_rank_3-5_1.ckpt
    output/checkpoint/rank_0/llama2_13b_rank_0-5_1.ckpt
    output/checkpoint/rank_5/llama2_13b_rank_5-5_1.ckpt
    ```

    2). 执行命令 `cat output/msrun_log/worker_0.log | grep 'Epoch:'` 查看已经训练的 step：

    ```text
    $ cat output/msrun_log/worker_0.log | grep 'Epoch:'
    2025-04-07 15:34:27,308 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    1/   19], loss: 10.649, per_step_time: 103328ms, lr: 0.0, overflow cond: False, loss_scale: 1.0, global_norm: [1 31049], train_throughput_per_npu: 2.896T
    2025-04-07 15:34:29,173 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    2/   19], loss: 10.633, per_step_time: 1752ms, lr: 1e-05, overflow cond: False, loss_scale: 1.0, global_norm: [1 508834], train_throughput_per_npu: 170.738T
    2025-04-07 15:34:30,941 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    3/   19], loss: 9.673, per_step_time: 1754ms, lr: 9.981987e-06, overflow cond: False, loss_scale: 1.0, global_norm [10.579812], train_throughput_per_npu: 170.523T
    2025-04-07 15:34:32,704 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    4/   19], loss: 9.287, per_step_time: 1756ms, lr: 9.928079e-06, overflow cond: False, loss_scale: 1.0, global_norm [21.932272], train_throughput_per_npu: 170.319T
    2025-04-07 15:34:34,469 - [mindformers/core/callback/callback.py:529] - INFO - { Epoch:[  1/  2], step:[    5/   19], loss: 8.867, per_step_time: 1758ms, lr: 9.8386645e-06, overflow cond: False, loss_scale: 1.0, global_norm [16.986555], train_throughput_per_npu: 170.173T
    ```

    3). 执行命令 `cat output/msrun_log/worker_0.log | grep 'report group list:'` 查看日志中 MindIO 输出的副本关系：

    ```text
    $ cat output/msrun_log/worker_0.log | grep 'report group list:'
    2025-04-07 15:34:27.363613 info 1879138 [TTP controller.cpp:1512] rank:4, report group list: [0, 4]
    2025-04-07 15:34:27.385564 info 1879139 [TTP controller.cpp:1512] rank:7, report group list: [3, 7]
    2025-04-07 15:34:27.393198 info 1879136 [TTP controller.cpp:1512] rank:6, report group list: [2, 6]
    2025-04-07 15:34:27.393515 info 1879142 [TTP controller.cpp:1512] rank:1, report group list: [1, 5]
    ```

    从上面训练的 step 信息可以看出已经训练的 5 个 step，和 Checkpoint 的文件名 `llama2_13b_rank_2-5_1.ckpt` 中的 5 是一致的。

    从日志中输出的副本关系 `[0, 4]`、`[3, 7]`、`[2, 6]` 和 `[1, 5]` 得知：

    - rank 0 和 rank 4 权重存在副本关系，临终的 Checkpoint 保存在 rank 0
    - rank 3 和 rank 7 权重存在副本关系，临终的 Checkpoint 保存在 rank 3
    - rank 2 和 rank 6 权重存在副本关系，临终的 Checkpoint 保存在 rank 2
    - rank 1 和 rank 5 权重存在副本关系，由于 worker 1 终止，临终的 Checkpoint 保存在 rank 5

### 故障快速恢复

本章节以 Llama3.1-8B 训练为例演示故障快速恢复的使用。

> 以下示例所展示的参数数值仅作为实验数据，请以真实训练数据为准。

1. 先安装 [MindSpore](https://www.mindspore.cn/install)。
2. 下载 MindSpore Transformers，使用的[finetune_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml)按照如下配置添加和修改参数：

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

    **参数说明：**

    | 参数名称                      | 描述                                              | 类型    | 是否可选     |
    |-----------------------------|-------------------------------------------------|-------|----------|
    | output_dir                  | 保存权重和切分策略的文件路径。默认值为`./output`。                  | str   | 可选       |
    | monitor_config              | 训练指标监控配置。默认值为`None`。                            | dict  | 可选       |
    | monitor_on                  | 是否开启训练指标监控配置。只有开启时才能监测异常的global norm和使能TRE功能。   | bool  | 必选`True` |
    | check_for_global_norm       | 是否开启进程级故障快速恢复功能，和数据跳过功能互斥。默认值为`False`。          | bool  | 可选       |
    | global_norm_spike_threshold | global norm的阈值，当global norm超过时触发数据跳过。默认值为`3.0`。 | float | 可选       |
    | callbacks                   | callbacks配置。                                    | list  | 必选       |
    | save_checkpoint_steps       | 保存权重的步数间隔。                                      | int   | 必选       |

3. 配置环境变量：

   ```shell
   export MS_ENABLE_TFT="TRE:1"
   ```

4. 运行以下命令，开启训练：

    ```shell
    cd mindformers

    bash scripts/msrun_launcher.sh "run_mindformer.py \
        --register_path research/llama3_1 \
        --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
        --train_data /{path}/wiki4096.mindrecord \
        --run_mode train \
        --use_parallel True" 8
    ```

5. 模型正式开始训练时，遇到global norm大于设定阈值，则会打印如下日志，提示用户当前遇到异常global norm，并记录对应的global step和global norm到abnormal_global_norm.json中，触发报错，进入快速恢复阶段。

    ```text
    - INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 11.905, per_step_time: 2775ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [45.702465], train_throughput_per_npu: 171.176T
    - INFO -    0.0% |                                                  | 0.36029 samples/s/p  10:01:16 }
    - INFO - Current global norm [45.702465] is greater equal than threshold 44.0, stop training...
    ```

6. 重新拉起训练后，从之前断点的步数开始续训。如果在训练至相同的global step时，global norm仍然大于设定的阈值，由于此前已经将对应的global step记录到YAML设置的output_dir下的abnormal_global_norm.json中，故此处只会记录相应的global norm，并不会抛出异常。

    ```text
    - INFO - { Epoch:[  1/  2], step:[    2/ 6500], loss: 11.905, per_step_time: 3504ms, lr: 2.5641025e-08, overflow cond: False, loss_scale: 1.0, global_norm: [45.706497], train_throughput_per_npu: 135.552T
    - INFO -    0.0% |                                                  | 0.28531 samples/s/p  12:39:17 }
    - INFO - The global norm [45.706497] of step 2 is still greater or equal than threshold 44.0, continue training.
    ```

    abnormal_global_norm.json记录数据如下：

    ```json
    {
      "2": [45.70246505737305, 45.70649719238281]
    }
    ```

    "2"表示对应训练步数的global step，后面列表记录的则是恢复前后训练的global norm。
