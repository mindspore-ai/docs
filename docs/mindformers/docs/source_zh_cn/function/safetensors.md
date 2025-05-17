# Safetensors权重

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/safetensors.md)

## 概述

Safetensors 是 Huggingface 推出的一种可靠、易移植的机器学习模型存储格式，用于安全地存储Tensor，而且存储速度较快（零拷贝）。本文主要介绍MindSpore Transformers如何支持该文件格式的保存与加载，帮助用户更好更快地使用权重。

## Safetensors权重示例

Safetensors文件主要分为两种类型：完整权重文件和分布式权重文件。以下是它们的获取方式及对应的文件示例。

### 完整权重

Safetensors完整权重可通过以下两种方式获取：

1. 直接从Huggingface上下载。
2. 通过MindSpore Transformers分布式训练后，通过[合并脚本](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html#safetensors%E6%9D%83%E9%87%8D%E7%A6%BB%E7%BA%BF%E5%90%88%E5%B9%B6)生成完整权重。

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

## 配置说明

加载相关配置：

| 参数名称              | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| load_checkpoint     | 预加载权重的文件夹路径。<br> - 如果是完整权重，填写切片/单个权重文件所在文件夹路径。<br/>注：支持Huggingface safetensor权重加载（当前仅支持Llama系列模型）。在线加载过程中，会保存一份转换后的MindSpore safetensor权重文件至`/output/ms_safetensors下`。<br> - 如果是分布式权重，需按照`model_dir/rank_x/xxx.safetensor`格式存放，文件夹路径填写为`model_dir`。                                                                                                                                                                                                             |
| load_ckpt_format | 加载的模型权重的格式，可选`ckpt`、`safetensors`，默认为`ckpt`。<br/>加载权重为`safetensors`格式时，需配套修改此配置为`safetensors`。                                                                                                                                                                                                                                                                                                                                                                  |
| auto_trans_ckpt | 是否开启在线切分功能。<br/>- 如果加载权重是完整权重：<br/>a. `use_parallel: True`时，判断为分布式加载，需同步设置`auto_trans_ckpt: True`，开启在线切分功能。<br/>b. `use_parallel: False`时，判断为单卡加载，需同步设置`auto_trans_ckpt: False`，关闭在线切分功能。<br/>- 如果加载权重是分布式权重：<br/>a. 不改变原有切分策略，需设置`auto_trans_ckpt: False`，直接按原先切分策略直接加载。<br/>b. 改变原有切分策略，需设置`auto_trans_ckpt: True` 并配置`src_strategy_path_or_dir`为原有切分策略文件路径。<br/>任务拉起时，会将权重在线合并为完整权重，并依据配置文件中设定的并行策略进行切分与加载。在线合并的完整权重会保存在当前目录`/output/unified_checkpoint`文件下。 |
| remove_redundancy | 加载的权重是否为去冗余后的权重，默认为`False`。                                                                                                                                                                                                                                                                                                                                                                                                                                 |

保存相关配置：

| 参数名称                    | 说明                                                         |
| :-------------------------- | ------------------------------------------------------------ |
| callbacks.checkpoint_format | 保存的模型权重的格式，默认值为`ckpt`。可选`ckpt`，`safetensors`。 |
| callbacks.remove_redundancy | 保存权重时是否开启去冗余保存功能，默认为`False`。仅支持`safetensors格式`。 |

## 使用示例

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

更多详情请参考：[预训练介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/pre_training.html)

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

更多详情请参考：[SFT微调介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/sft_tuning.html)

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

更多详情请参考：[推理介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/usage/inference.html)

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

大集群规模场景下，避免在线合并过程耗时过长占用训练资源，推荐将原分布式权重文件离线[合并完整权重](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html#safetensors%E6%9D%83%E9%87%8D%E7%A6%BB%E7%BA%BF%E5%90%88%E5%B9%B6)后传入，无需传入源切分策略文件路径。

更多详情请参考：[断点续训介绍](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/resume_training.html)。

## 权重保存

### 概述

在深度学习模型的训练过程中，保存模型的权重是至关重要的一步。权重保存功能使得我们能够在训练的任意阶段存储模型的参数，以便用户在训练中断或完成后进行恢复、继续训练、评估或部署。同时还可以通过保存权重的方式，在不同环境下复现实验结果。

目前，MindSpore TransFormer 支持 [safetensors](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/safetensors.html) 格式的权重文件读取和保存。

### 目录结构

在训练过程中，MindSpore Transformers 默认会在输出目录（同训练日志，默认为 `./output` ）中生成权重保存文件夹： `checkpoint` 。

如果在 yaml 中设置了配置项 `save_network_params: True` 后，会额外生成权重保存文件夹 `checkpoint_network` 。

| 文件夹                | 描述                                                        |
|--------------------|-----------------------------------------------------------|
| checkpoint         | 保存模型权重、优化器状态、step 和 epoch 于 safetensors 文件中，可用于**断点恢复训练**。 |
| checkpoint_network | 仅保存模型权重参数于 safetensors 文件中，适用于后续进行微调、推理、评测，不支持断点续训。       |

#### `checkpoint`目录结构

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

##### 权重相关文件说明

| 文件                                  | 描述                                                                                                                                                                                                                                                                        |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| meta.json                           | 记录最后保存的权重的 `epoch` 、 `step` 和权重名，每个 rank 进程独立维护一个 `meta.json` 文件。                                                                                                                                                                                                         |
| {prefix}-{epoch}_{step}.safetensors | 保存的权重文件， `prefix` 包含 rank_id 信息，格式为 `{prefix}-{epoch}_{step}.safetensors` 。如果前缀相同的文件已经存在，系统会自动递增后缀。<br>开启数据下沉时， `epoch` 位置计算方式为 $\frac{CurrentTotalStepNumber}{SinkSize} = \frac{((CurrentEpoch-1)*StepsPerEpoch+CurrentStepInEpoch)}{SinkSize}$，`step` 固定为 `sink_size` 。 |

### 配置与使用

#### YAML参数配置

用户可通过修改配置文件来控制权重保存的行为。以下是主要参数：

用户可修改 `yaml` 配置文件中 `CheckpointMonitor` 下的字段来控制权重保存行为。

以 [`DeepSeek-V3` 预训练 yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L206) 为例，可做如下配置：

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

##### 主要配置参数介绍

有关保存权重配置的主要参数如下表所列：

| 参数                    | 描述                                                      | 取值说明                                                                                                          |
|-----------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| prefix                | 模型权重文件的前缀名，可用于指代模型名字。                                   | (str, 可选) - 默认值： `"CKP"` 。                                                                                    |
| save_checkpoint_steps | 每训练多少步保存一次权重。                                           | (int, 可选) - 默认值： `1` ，不设置时不保存模型权重。                                                                            |
| keep_checkpoint_max   | 最多同时保存多少个权重文件，达到上限后会在保存权重时删除最旧的权重文件。                    | (int, 可选) - 默认值： `5` ，不设置时不对文件夹下权重数量进行监控和删除。                                                                  |
| integrated_save       | 在并行场景下是否合并保存拆分的 Tensor。合并保存功能仅支持在自动并行场景中使用，在手动并行场景中不支持。 | (bool, 可选) - 默认值： `False`                                                                                     |
| async_save            | 是否使用异步方式保存 safetensors 文件。                              | (bool, 可选) - `True` 时默认使用异步线程，默认值： `False` 。                                                                  |
| checkpoint_format     | 输出文件的格式，需要配置为 `safetensors` 。                           | (str, 可选) - 模型权重保存的格式。支持 `"ckpt"` 、 `"safetensors"` 。默认值： `ckpt` 。（注意： ckpt 格式将在后续版本中日落，推荐使用 safetensors 格式。） |
| remove_redundancy     | 保存模型权重时是否去除冗余。                                          | (bool, 可选) - 默认值： `False` 。                                                                                   |
| save_network_params   | 是否仅额外保存网络参数。                                            | (bool, 可选) - 是否仅额外保存网络参数。默认值： `False` 。                                                                       |

如果您想了解更多有关 CheckpointMonitor 的知识，可以关注 [CheckpointMonitor API 文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/core/mindformers.core.CheckpointMonitor.html)。
