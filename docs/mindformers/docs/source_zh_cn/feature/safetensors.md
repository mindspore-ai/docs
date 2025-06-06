# Safetensors权重

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/safetensors.md)

## 概述

Safetensors 是 Huggingface 推出的一种可靠、易移植的机器学习模型存储格式，用于安全地存储Tensor，而且存储速度较快（零拷贝）。
本文主要介绍了safetensor的几种格式类型，以及MindSpore Transformers如何支持该格式权重的保存与加载，权重特性，权重的分布式切分与合并和权重格式转换，帮助用户更好更快地使用权重。

## 权重示例

Safetensors文件主要分为两种类型：完整权重文件和分布式权重文件。以下是它们的获取方式及对应的文件示例。

### 完整权重

Safetensors完整权重可通过以下两种方式获取：

1. 直接从Huggingface上下载。
2. 通过MindSpore Transformers分布式训练后，通过[合并脚本](#权重合并)生成完整权重。

Huggingface Safetensors示例目录结构：

```text
qwen2_7b
 └── hf_unified_safetenosrs
        ├── model-00001-of-00004.safetensors
        ├── model-00002-of-00004.safetensors
        ├── model-00003-of-00004.safetensors
        ├── model-00004-of-00004.safetensors
        └── model.safetensors.index.json        # Huggingface权重参数和文件的存储关系映射json文件
```

MindSpore Safetensors示例目录结构：

```text
qwen2_7b
 └── ms_unified_safetenosrs
        ├── model-00001-of-00004.safetensors
        ├── model-00002-of-00004.safetensors
        ├── model-00003-of-00004.safetensors
        ├── model-00004-of-00004.safetensors
        ├── hyper_param.safetensors            # 训练任务记录的超参文件
        └── param_name_map.json                # MindSpore权重参数和文件的存储关系映射json文件
```

### 分布式权重

Safetensors分布式权重可通过以下两种方式获取：

1. 通过MindSpore Transformers分布式训练生成。
2. 通过[格式转换脚本](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.ckpt_to_safetensors.html)，将原有分布式ckpt权重转换为Safetensors格式。

分布式Safetensors示例目录结构：

```text
qwen2_7b
 └── distributed_safetenosrs
        ├── rank_0
            └── qwen2_7b_rank_0.safetensors
        ├── rank_1
            └── qwen2_7b_rank_1.safetensors
        ...
        └── rank_x
            └── qwen2_7b_rank_x.safetensors
```

## 权重保存

### 概述

在深度学习模型的训练过程中，保存模型的权重是至关重要的一步。权重保存功能使得我们能够在训练的任意阶段存储模型的参数，以便用户在训练中断或完成后进行恢复、继续训练、评估或部署。同时还可以通过保存权重的方式，在不同环境下复现实验结果。

目前，MindSpore Transformers 支持 [safetensors](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html) 格式的权重文件读取和保存。

### 目录结构

在训练过程中，MindSpore Transformers 默认会在输出目录（同训练日志，默认为 `./output` ）中生成权重保存文件夹： `checkpoint` 。

如果在 yaml 中设置了配置项 `save_network_params: True` 后，会额外生成权重保存文件夹 `checkpoint_network` 。

| 文件夹             | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| checkpoint         | 保存模型权重、优化器状态、step 和 epoch 于 safetensors 文件中，可用于**断点恢复训练**。 |
| checkpoint_network | 仅保存模型权重参数于 safetensors 文件中，适用于后续进行微调、推理、评测，不支持断点续训。 |

#### checkpoint目录结构

以一个 8 卡任务为例，`output` 文件夹中的权重文件按如下格式保存：

```text
output
    ├── checkpoint
        ├── rank_0
            ├── meta.json
            └── {prefix}-{epoch}_{step}.ckpt
        ...
        └── rank_7
            ├── meta.json
            └── {prefix}-{epoch}_{step}.ckpt
    └──checkpoint_network
        ├── rank_0
            └── {prefix}-{epoch}_{step}.safetensors
        ...
        └── rank_7
            └── {prefix}-{epoch}_{step}.safetensors
```

权重相关文件说明

| 文件                                | 描述                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| meta.json                           | 记录最后保存的权重的 `epoch` 、 `step` 和权重名，每个 rank 进程独立维护一个 `meta.json` 文件。 |
| {prefix}-{epoch}_{step}.safetensors | 保存的权重文件， `prefix` 包含 rank_id 信息，格式为 `{prefix}-{epoch}_{step}.safetensors` 。如果前缀相同的文件已经存在，系统会自动递增后缀。<br>开启数据下沉时， `epoch` 位置计算方式为 $\frac{CurrentTotalStepNumber}{SinkSize} = \frac{((CurrentEpoch-1)*StepsPerEpoch+CurrentStepInEpoch)}{SinkSize}$，`step` 固定为 `sink_size` 。 |

### 配置与使用

#### YAML参数配置

用户可通过修改配置文件来控制权重保存的行为。以下是主要参数：

用户可修改 `yaml` 配置文件中 `CheckpointMonitor` 下的字段来控制权重保存行为。

以 [DeepSeek-V3 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L206) 为例，可做如下配置：

```yaml
# callbacks
callbacks:
  ...
  - type: CheckpointMonitor
    prefix: "deepseekv3"
    save_checkpoint_steps: 1000
    integrated_save: False
    async_save: False
    checkpoint_format: "safetensors"
  ...
```

该配置的含义为：每隔 1000 步保存一次 safetensors 权重、最多同时存储 5 个权重、并行场景下不合并保存拆分的 Tensor、且不使用异步方式保存权重文件。

有关保存权重配置的主要参数如下表所列：

| 参数                  | 描述                                                         | 取值说明                                                     |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| prefix                | 模型权重文件的前缀名，可用于指代模型名字。                   | (str, 可选) - 默认值： `"CKP"` 。                            |
| save_checkpoint_steps | 每训练多少步保存一次权重。                                   | (int, 可选) - 默认值： `1` ，不设置时不保存模型权重。        |
| keep_checkpoint_max   | 最多同时保存多少个权重文件，达到上限后会在保存权重时删除最旧的权重文件。 | (int, 可选) - 默认值： `5` ，不设置时不对文件夹下权重数量进行监控和删除。 |
| integrated_save       | 在并行场景下是否合并保存拆分的 Tensor。合并保存功能仅支持在自动并行场景中使用，在手动并行场景中不支持。 | (bool, 可选) - 默认值： `False`                              |
| async_save            | 是否使用异步方式保存 safetensors 文件。                      | (bool, 可选) - `True` 时默认使用异步线程，默认值： `False` 。 |
| checkpoint_format     | 输出文件的格式，需要配置为 `safetensors` 。                  | (str, 可选) - 模型权重保存的格式。支持 `"ckpt"` 、 `"safetensors"` 。默认值： `ckpt` 。（注意： ckpt 格式将在后续版本中日落，推荐使用 safetensors 格式。） |
| remove_redundancy     | 保存模型权重时是否去除冗余。                                 | (bool, 可选) - 默认值： `False` 。                           |
| save_network_params   | 是否仅额外保存网络参数。                                     | (bool, 可选) - 是否仅额外保存网络参数。默认值： `False` 。   |

如果您想了解更多有关 CheckpointMonitor 的知识，可以参考 [CheckpointMonitor API 文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CheckpointMonitor.html)。

## 权重加载

### 概述

MindSpore Transformers支持训练、推理、续训在单卡多卡全场景下的权重加载，包括完整权重和分布式权重。可参考以下说明，针对相应场景调整配置。

### 配置说明

| 参数名称         | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| load_checkpoint  | 预加载权重的文件夹路径。<br> - 如果是完整权重，填写切片/单个权重文件所在文件夹路径。<br/>注：支持Huggingface safetensor权重加载（当前仅支持Llama系列模型）。在线加载过程中，会保存一份转换后的MindSpore safetensor权重文件至`/output/ms_safetensors`下。<br> - 如果是分布式权重，需按照`model_dir/rank_x/xxx.safetensor`格式存放，文件夹路径填写为`model_dir`。 |
| load_ckpt_format | 加载的模型权重的格式，可选`ckpt`、`safetensors`，默认为`ckpt`。<br/>加载权重为`safetensors`格式时，需配套修改此配置为`safetensors`。 |
| use_parallel     | 是否并行加载。                                               |
| auto_trans_ckpt  | 是否开启在线切分功能。<br/>- 如果加载权重是完整权重：<br/>a. `use_parallel: True`时，判断为分布式加载，需同步设置`auto_trans_ckpt: True`，开启在线切分功能。<br/>b. `use_parallel: False`时，判断为单卡加载，需同步设置`auto_trans_ckpt: False`，关闭在线切分功能。<br/>- 如果加载权重是分布式权重：<br/>a. 不改变原有切分策略，需设置`auto_trans_ckpt: False`，直接按原先切分策略直接加载。<br/>b. 改变原有切分策略，需设置`auto_trans_ckpt: True` 并配置`src_strategy_path_or_dir`为原有切分策略文件路径。<br/>任务拉起时，会将权重在线合并为完整权重，并依据配置文件中设定的并行策略进行切分与加载。在线合并的完整权重会保存在当前目录`/output/unified_checkpoint`文件下。 |

### 完整权重加载

#### 单卡加载

```yaml
# 配置文件
load_checkpoint: '/qwen2_7b/unified_safetenosrs'    # 加载完整权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: False                              # 完整权重+单卡加载时需关闭此配置项
use_parallel: False                                 # 单卡加载
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
```

#### 多卡加载

```yaml
# 配置文件
load_checkpoint: '/qwen2_7b/unified_safetenosrs'    # 加载完整权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 完整权重+分布式加载时需打开此配置项，开启在线切分功能
use_parallel: True                                  # 多卡加载
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
```

### 分布式权重加载

#### 多卡加载-原有切分策略

```yaml
# 配置文件
load_checkpoint: '/output/distributed_safetenosrs'  # 加载源分布式权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: False                              # 关闭在线切分功能
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 1
```

#### 多卡加载-改变切分策略

```yaml
# 配置文件
load_checkpoint: '/output/distributed_safetenosrs'  # 加载源分布式权重文件路径
src_strategy_path_or_dir: '/output/src_strategy'    # 加载源策略文件，用于合并源分布式权重为完整权重
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 开启在线切分功能
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 4
  model_parallel: 2
  pipeline_stage: 1
```

大集群规模场景下，避免在线合并过程耗时过长占用训练资源，推荐将原分布式权重文件离线[合并完整权重](#权重合并)后传入，此时无需传入源切分策略文件路径。

### 特殊场景

#### 物理机多机多卡训练

大规模模型通常需要通过多台服务器组成的集群进行训练。权重切分转换需要依赖编译完成后的目标切分策略文件，在这种多机多卡的场景下，如果服务器之间存在共享盘，生成的策略文件在同一个目录下，则可以使用自动转换功能；如果服务器之间无共享盘，需要手动复制策略文件后在进行转换功能。下面以两台服务器、16卡训练为例进行说明。

**场景一：服务器之间有共享盘**

在服务器之间有共享盘的场景下，可以使用 MindSpore Transformers 的自动权重转换功能在多机多卡训练之前自动进行权重转换。假设 `/data` 为服务器的共享盘，且 MindSpore Transformers 的工程代码位于 `/data/mindformers` 路径下。

**参数配置：**

```yaml
output_dir: './output'                              # 策略文件会生成在./output/strategy下，用于权重在线切分
load_checkpoint: '/qwen2_7b/unified_safetenosrs'    # 加载完整权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 完整权重+分布式加载时需打开此配置项，开启在线切分功能
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/worker/dataset/wiki103/"
    shuffle: True
parallel_config:                                    # 配置16卡分布式策略（仅供参考）
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 2
  vocab_emb_dp: True
  gradient_aggregation_group: 4
  micro_batch_interleave_num: 1
```

**启动任务**：

使用[mindformers/scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh)进行任务启动。

  ```shell
  # 第一台服务器（主节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 0 output/msrun_log False 300
  # 第二台服务器（子节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 1 output/msrun_log False 300
  ```

**场景二：服务器之间无共享盘**

在服务器之间无共享盘的情况下，需要对生成的策略文件进行离线合并和转发操作后再使能在线切分功能。以下步骤描述了如何进行该操作，并启动多机多卡训练任务。

**1.获取分布式策略**

在进行离线权重转换前，首先需要获取各节点的分布式策略文件。

```yaml
  # 设置 only_save_strategy 为 True 以获取分布式策略文件，生成后任务自动退出
  only_save_strategy: True

  # 配置数据集路径
  train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "/worker/dataset/wikitext_2048/"
      shuffle: True

  # 配置16卡分布式策略（仅供参考）
  parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 2
    micro_batch_num: 2
    vocab_emb_dp: True
    gradient_aggregation_group: 4
    micro_batch_interleave_num: 1
```

各节点的策略文件将分别保存在各自的`output/strategy`目录中。例如，节点0仅保存`ckpt_strategy_rank_0-7.ckpt`文件，节点1仅保存`ckpt_strategy_rank_8-15.ckpt`文件。随后，需将所有节点的策略文件集中到同一台服务器上，以便进行后续操作，集中后的目录及文件如下。

```text
output
    ├── strategy
        ├── ckpt_strategy_rank_0.ckpt
        ...
        ├── ckpt_strategy_rank_7.ckpt
        ├── ckpt_strategy_rank_8.ckpt
        ...
        └── ckpt_strategy_rank_15.ckpt
```

**2.合并分布式策略**

调用MindSpore提供的[策略合并接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.merge_pipeline_strategys.html)将集中后的所有策略文件合并成一个文件，用于后续权重切分。

```python
import mindspore as ms
ms.parallel.merge_pipeline_strategys("/output/strategy", "/output/merged_strategy/dst_strategy.ckpt")
```

**3.权重切分加载**

**分发策略文件+在线切分（推荐）：**

将合并后的策略文件`dst_strategy.ckpt`分发到各个节点下的`./output/merged_strategy/`目录下，打开自动切分功能，重新拉起训练任务。每个节点的配置文件均需要修改。

```yaml
output_dir: './output'                              # 确保每个节点下的./output/merged_strategy/都有合并完后的策略文件
load_checkpoint: '/qwen2_7b/unified_safetenosrs'    # 加载完整权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 完整权重+分布式加载时需打开此配置项，开启在线切分功能
```

**离线切分+分发分布式权重：**

根据[权重切分](#权重切分)指南，先将完整权重离线切分成分布式权重文件，再分发到各台机器，关闭自动切分功能，配置`load_checkpoint`为分布式权重路径。每个节点的配置文件均需要修改。

因为分布式权重文件一般比策略文件大，分发操作更耗时，更推荐第一种方式。

```yaml
load_checkpoint: '/output/distributed_safetenosrs'  # 加载分布式权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: False                              # 分布式权重加载，关闭在线切分功能
```

**4.启动任务**：

使用[mindformers/scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh)进行任务启动。

  ```shell
  # 第一台服务器（主节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 0 output/msrun_log False 300
  # 第二台服务器（子节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 1 output/msrun_log False 300
  ```

## 权重特性

### 去冗余保存及加载

当前MindSpore Transformers保存权重时，默认会在dp/opt域重复保存多份一致的权重文件，导致带来额外的存储开销和负担。可通过以下的配置和使用方法，实现dp/opt去冗余保存和加载，有效降低千卡及以上大规模集群下的存储压力。此特性仅在分布式权重下生效，完整权重不涉及去冗余。

保存时打开以下配置：

```yaml
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                  # 保存权重文件格式
    remove_redundancy: True                         # 保存权重时开启去冗余
```

保存后的分布式权重大小不同，总权重文件小于去冗余功能开启前：

```text
output
    ├── checkpoint
        ├── rank_0
            └── example-1_1.ckpt  #文件大小：5.2G
        ├── rank_1
            └── example-1_1.ckpt  #文件大小：5.2G
        ...
        ├── rank_6
            └── example-1_1.ckpt  #文件大小：4.1G
        └── rank_7
            └── example-1_1.ckpt  #文件大小：4.1G
```

加载时打开以下配置：

```yaml
load_ckpt_format: 'safetensors'    # 加载权重文件格式
remove_redundancy: True            # 加载权重时开启去冗余
```

> MindSpore Transformers 1.5.0及以下版本当去冗余保存和加载的配置项不一致时，可能导致精度异常，请确保配置正确。1.5.0以上版本将根据传入的权重是否去冗余自动识别并加载，无需关注加载配置。

## 权重切分与合并

### 概述

在当前的分布式训练和推理环境中，当用户需要改变分布式策略时，需要先将已有的分布式权重合并成完整权重后，再通过在线切分/离线切分的方式完成权重加载。为满足不同场景下的权重转换需求，可以参考下面脚本和接口，实现权重多卡合并单卡和单卡切分多卡的功能。

### 权重合并

#### 使用说明

使用MindSpore Transformers提供的[safetensors权重合并脚本](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/safetensors/unified_safetensors.py)，按照如下方式进行safetensors权重合并。合并后的权重格式为[完整权重](#完整权重)。

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy has_redundancy
```

#### 参数说明

- **src_strategy_dirs**：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 `output/strategy/` 目录下。分布式权重需根据以下情况填写：

    - **源权重开启了流水线并行**：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 `ckpt_strategy_rank_x.ckpt` 文件合并，并在文件夹下生成 `merged_ckpt_strategy.ckpt`。如果已经存在 `merged_ckpt_strategy.ckpt`，可以直接填写该文件的路径。
    - **源权重未开启流水线并行**：权重转换可基于任一策略文件，填写任意一个 `ckpt_strategy_rank_x.ckpt` 文件的路径即可。

    **注意**：如果策略文件夹下已存在 `merged_ckpt_strategy.ckpt` 且仍传入文件夹路径，脚本会首先删除旧的 `merged_ckpt_strategy.ckpt`，再合并生成新的 `merged_ckpt_strategy.ckpt` 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。
- **mindspore_ckpt_dir**：分布式权重路径，请填写源权重所在文件夹的路径，源权重应按 `model_dir/rank_x/xxx.safetensors` 格式存放，并将文件夹路径填写为 `model_dir`。
- **output_dir**：目标权重的保存路径，默认值为 "/new_llm_data/******/ckpt/nbg3_31b/tmp"，即目标权重将放置在 `/new_llm_data/******/ckpt/nbg3_31b/tmp` 目录下。
- **file_suffix**：目标权重文件的命名后缀，默认值为 "1_1"，即目标权重将按照 `*1_1.safetensors` 格式查找。
- **has_redundancy**：合并的源权重是否是冗余的权重，默认为 `True`。
- **filter_out_param_prefix**：合并权重时可自定义过滤掉部分参数，过滤规则以前缀名匹配。如优化器参数"adam_"。
- **max_process_num**：合并最大进程数。默认值：64。

#### 示例

场景一：

如果合并去除冗余的safetensors权重，可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy False
```

场景二：

如果合并过滤Adam优化器的safetensors权重，可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --filter_out_param_prefix "adam_"
```

### 权重切分

#### 使用说明

使用MindSpore提供的[策略合并接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.merge_pipeline_strategys.html)和[切分保存接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.load_distributed_checkpoint.html)，按照如下方式进行safetensors权重离线切分保存。切分后的权重格式为[分布式权重](#分布式权重)。

```python
import mindspore as ms
# step1:合并目标切分策略文件
ms.parallel.merge_pipeline_strategys("/output/strategy", "/output/merged_strategy/dst_strategy.ckpt")
# step2:根据合并后的目标切分策略以及完整权重，将权重切分并保存成分布式权重
ms.load_distributed_checkpoint(
            network=None,
            predict_strategy='/output/merged_strategy/dst_strategy.ckpt',
            unified_safetensors_dir='/path/unified_safetensors',
            dst_safetensors_dir='/path/distributed_safetensors',
            format='safetensors',
            max_process_num=64
        )
```

#### 参数说明

- **network** (Cell) - 分布式预测网络，format为 safetensors 时，network传递为None，此时接口执行保存模式。
- **predict_strategy** (Union[dict, str]) - 目标切分策略文件。默认值： `None` 。
- **unified_safetensors_dir** (str) - 完整权重文件目录。默认值： `None` 。
- **dst_safetensors_dir** (str) - 保存模式场景下，权重的保存目录。
- **max_process_num** (int) - 最大进程数。默认值：64。

## 权重格式转换

### Ckpt转换Safetensors

MindSpore Transformers存量权重文件为ckpt格式，可以通过以下两种方式实现格式转换成safetensors文件。

#### 接口调用

直接调用[Mindspore格式转换接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.ckpt_to_safetensors.html)实现。

```python
import mindspore as ms
ms.ckpt_to_safetensors("./ckpt_save_path/rank0/checkpoint_0.ckpt", ".output/safetensors_path/")
#参数说明
#file_path (str) - 包含 checkpoint 文件的目录路径或单个 checkpoint 文件 (.ckpt) 的路径
#save_path (str, 可选) - 保存 safetensors 文件的目录路径。默认值：None
```

#### 训练任务

调整配置文件后启动MindSpore Transformers训练任务，通过以ckpt格式加载和safetensor格式保存的方法实现转换。

```yaml
load_checkpoint: 'output/checkpoint/'               # 加载权重文件路径
load_ckpt_format: 'ckpt'                            # 加载权重文件格式为ckpt
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: 'safetensors'                # 保存权重文件格式为safetensor
```

## 任务示例

### 预训练任务示例

以Llama2-7B为例，修改配置项[pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml)确认权重保存格式：

```yaml
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                  # 保存权重文件格式
    remove_redundancy: True                         # 保存权重时开启去冗余
```

完成后执行命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/pretrain_llama2_7b.yaml \
 --train_dataset_dir /{path}/wiki4096.mindrecord \
 --use_parallel True \
 --run_mode train" 8
```

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

更多详情请参考：[预训练介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html)

### 微调任务示例

若使用完整权重多卡在线微调，以Qwen2-7B模型为例，修改配置项[finetune_qwen2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/qwen2_7b/finetune_qwen2_7b.yaml)：

```yaml
# 修改后的配置
load_checkpoint: '/qwen2_7b/hf_unified_safetenosrs' # 加载权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 完整权重时需打开此配置项，开启在线切分功能
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                  # 保存权重文件格式
```

若使用分布式权重多卡在线微调，以Qwen2-7B模型为例，修改配置项[finetune_qwen2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/qwen2_7b/finetune_qwen2_7b.yaml)：

```yaml
# 修改后的配置
load_checkpoint: '/qwen2_7b/distributed_safetenosrs' # 加载权重文件路径
load_ckpt_format: 'safetensors'                      # 加载权重文件格式
parallel_config:                                     # 配置目标分布式策略
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                  # 保存权重文件格式
```

完成后执行命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config research/qwen2/qwen2_7b/finetune_qwen2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-data.mindrecord \
 --register_path research/qwen2 \
 --use_parallel True \
 --run_mode finetune" 2
```

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

更多详情请参考：[SFT微调介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/supervised_fine_tuning.html)

### 推理任务示例

若使用完整权重多卡在线推理，以Qwen2-7B模型为例，修改配置项[predict_qwen2_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/qwen2_7b/predict_qwen2_7b_instruct.yaml)：

```yaml
# 修改后的配置
load_checkpoint: '/qwen2_7b/hf_unified_safetenosrs' # 加载权重文件路径
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 完整权重时需打开此配置项，开启在线切分功能
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
```

若使用分布式权重多卡在线推理，以Qwen2-7B模型为例，修改配置项[predict_qwen2_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/qwen2_7b/predict_qwen2_7b_instruct.yaml)：

```yaml
# 修改后的配置
load_checkpoint: '/qwen2_7b/distributed_safetenosrs' # 加载权重文件路径
load_ckpt_format: 'safetensors'                      # 加载权重文件格式
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
```

完成后执行命令：

```shell
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config research/qwen2/qwen2_7b/predict_qwen2_7b_instruct.yaml \
--run_mode predict \
--use_parallel True \
--register_path research/qwen2 \
--predict_data 'I love Beijing, because'" \
2
```

执行以上单卡推理和多卡推理命令的结果如下：

```text
'text_generation_text': [I love Beijing, because it is a city with a long history and culture.......]
```

更多详情请参考：[推理介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/inference.html)

### 断点续训任务示例

MindSpore Transformers支持step级断点续训功能，允许在训练中保存模型的checkpoint，并在训练中断后，加载保存的checkpoint恢复之前的状态继续训练。

若使用分布式权重多卡续训且不改变切分策略，修改配置项后启动原训练任务：

```yaml
# 修改后的配置
load_checkpoint: '/output/checkpoint'                # 加载源分布式权重文件路径
load_ckpt_format: 'safetensors'                      # 加载权重文件格式
resume_training: True                                # 断点续训功能开关
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                   # 保存权重文件格式
```

若分布式权重多卡续训且改变切分策略，需额外传入源切分策略文件路径，修改配置项后启动原训练任务：

```yaml
# 修改后的配置
load_checkpoint: '/output/checkpoint'               # 加载源分布式权重文件路径
src_strategy_path_or_dir: '/output/src_strategy'    # 加载源策略文件，用于合并源分布式权重为完整权重
load_ckpt_format: 'safetensors'                     # 加载权重文件格式
auto_trans_ckpt: True                               # 开启在线切分功能
resume_training: True                               # 断点续训功能开关
parallel_config:                                    # 配置目标分布式策略
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 1
callbacks:
  - type: CheckpointMonitor
    checkpoint_format: safetensors                  # 保存权重文件格式
```

更多详情请参考：[断点续训介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/resume_training.html)。

