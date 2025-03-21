# 高可用特性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/high_availability.md)

## 概述

MindSpore Transformers 高可用特性提供了如下三个功能：

- **临终 CKPT 功能**：主要针对大模型训练过程中的故障恢复加速，该特性在训练过程中发生故障后，校验中间状态数据的完整性和一致性，生成一次临终 CheckPoint 数据，恢复训练时能够通过该 CheckPoint 数据恢复，减少故障造成的训练迭代损失。
- **UCE 故障容错恢复功能**：主要是针对大模型训练过程中片上内存的 UCE 故障检测，并完成在线修复，达到 Step 级重计算。
- **进程级重调度恢复功能**：训练发生异常后，不需要重新拉起整个集群，只需以节点为单位进行重启或替换，完成修复并继续训练。

高可用特性目前只支持 MindSpore Ascend 后端的图模式；该特性同时需要支持Step级别恢复，因此配置数据下沉时只支持sink_size 为 1。

高可用特性的基础是两张卡存在副本关系，这样当其中一张卡发生故障时，可从另外一张卡恢复，因此权重和优化器都会存在两份冗余，会占用更多的显存。为保证这种冗余关系，必须开启数据并行，保证有两张卡权重一致，同时如果开启了优化器并行，也必须确保存在两张卡的优化器状态一致。

三个功能可同时开启，也可以单独开启。组合开启这三个功能时，依次生效的顺序是：UCE故障容错恢复 -> 进程级重调度恢复 -> 临终CKPT，如果其中一个功能可以恢复，就不会执行下一个功能。临终CKPT功能作为最后的保障，完成该功能后整个训练进程会退出，所以在另外两个功能开启时会默认开启。

临终CKPT保存Checkpoint文件以及通过该文件进行续训均使用现有MindSpore Transformers的能力，在使用方式上一致，只是临终CKPT依赖于strategy文件，因此在训练和续训时均需要配置该文件夹。

当异常触发临终的 CKPT 保存时，如果未开启去冗余保存，每个数据并行域只有一张卡保存了 CKPT，其余卡不会保存 CKPT；所以在恢复训练时，同样需要使能高可用特性才能恢复，否则其他卡无法找到可用的 CKPT，会报错退出。用户可通过计算分布式保存的CKPT数量是否为小于集群数量，来判断该CKPT是否由临终CKPT功能触发。

## 使用说明

高可用特性开关由环境变量使能，YAML 配置文件中不单独设置开关，但 YMAL 文件需要能配置出两张卡的权重和优化器状态一致。

高可用特性依赖用户安装 MindIO TFT SDK 包，详细请参考[在计算节点安装 MindIO TFT SDK](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft011.html)。

### 环境变量配置

```shell
export MINDIO_FOR_MINDSPORE=1
export MS_ENABLE_TFT="{TTP:1,UCE:1,ARF:1}"
export MS_TFT_IP=127.0.0.1
export MS_TFT_PORT=30051
```

- `MINDIO_FOR_MINDSPORE`：使能 MindIO TFT SDK 支持 MindSpore
- `MS_ENABLE_TFT`：表示启用 TTP、UCE 和 ARF 功能，如果只想启用其中的某一个功能，则将对应的值设置为 1 即可。
    - **TTP (Try To Persist)**：临终 CKPT 功能
    - **UCE (Uncorrectable Memory Error)**：UCE 故障容错恢复功能
    - **ARF (Air Refuelling)**：进程级重调度恢复功能
    - 开启 UCE 或者 ARF 功能时，默认开启 TTP 功能

- `MS_TFT_IP` 和 `MS_TFT_PORT` 分别表示 TFT Controller 的 IP 和端口号，无默认值，需要用户指定。如果由 MindSpore Transformers 启动 Controller，则配置用户集群中 rank0 节点的 IP 和端口号。如果用户自行启动 Controller，则配置 Controller 的 IP 和端口号。

### YAML 配置

YAML配置包含两部分：临终CKPT的保存及恢复配置和高可用的副本关系配置。

#### 保存及恢复配置

临终的CKPT保存和恢复能力分别用于初始训练和续训，这部分复用现有的MindSpore Transformers的配置，以下分别介绍初始训练和续训的配置。

- **初始训练配置**

    ```yaml
    output_dir: './output' # 保存CKPT和Strategy的目录
    load_checkpoint: ''    # 初次训练时配置为空
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: False  # 初次训练配置为False
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
    output_dir: './output' # 保存CKPT和Strategy的目录
    load_checkpoint: './output/checkpoint/'   # 续训时配置CKPT路径
    src_strategy_path_or_dir: '/output/strategy/'
    only_save_strategy: False
    resume_training: True  # 续训时配置为True
    run_mode: 'train'

    callbacks:
      - type: CheckpointMonitor
        prefix: "llama2_13b"
        save_checkpoint_steps: 100
        integrated_save: False
        async_save: False
    ```

#### 副本关系配置

高可用的三个功能的关键是配置出权重和优化器的副本冗余关系，配置的核心是数据并行域的维度大于2，如果叠加优化器并行，需要同时保证优化器的副本数大于2。所以配置分两类，开优化器并行和不开优化器并行。下面以8卡为例，介绍如何配置。

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

    开优化器并行后必须要保证优化器的状态存在副本，配置的关键是 optimizer_weight_shard_size 为 2。此时优化器状态的副本数为 data_parallel/optimizer_weight_shard_size。因此，如果数据并行度配置为2时，是不存在优化器副本的，必须把数据并行度配置为4；此时的副本数为 data_parallel/optimizer_weight_shard_size = 4/2 = 2。

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
