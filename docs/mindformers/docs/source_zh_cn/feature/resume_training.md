# 断点续训

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/resume_training.md)

## 概述

MindSpore Transformers支持**step级断点续训**功能，支持加载已保存的checkpoint来恢复之前的状态继续训练。这一特性在处理大规模训练任务时尤为重要，能够有效减少因意外中断导致的时间和资源浪费。

MindSpore Transformers支持保存和加载**ckpt**、**safetensors**两种格式权重，支持**中断续训**、**策略转换续训**、**增量续训**、**自动恢复续训**等多种续训场景，以及支持**加载最后保存完整的权重**、**加载指定step权重**、**加载MindSpore合并的权重**续训等不同的权重加载方式。

分布式环境中，断点续训要求所有节点的权重在**同一共享目录**下。用户可通过环境变量`SHARED_PATHS`来设置共享路径。

## 权重和策略文件介绍

MindSpore Transformers保存权重和策略文件，默认保存在`output/checkpoint`和`output/strategy`两个文件夹下，用户可以修改yaml配置的`output_dir`参数修改`output`文件夹路径。

权重文件主要保存了**网络参数**、**优化器参数**和**续训信息**，权重文件根据rank文件夹分开保存，每个rank文件夹下单独维护一个`meta.json`文件用以记录当前rank最后保存完整的权重信息。以单机8卡为例，权重保存格式如下：

```text
output/checkpoint
    ├── rank_0
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
    ├── rank_1
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
    ...
    ├── rank_7
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
```

> 权重名的prefix中携带rank_id信息，如：llama3_1_8b_rank_0；若保存权重时已存在相同prefix的权重，prefix会自动添加自增后缀以防止旧权重被覆盖。如"llama3_1_8b_rank_0"已存在时，prefix会更新为"llama3_1_8b_rank_0_1"，若"llama3_1_8b_rank_0_1"也已存在，prefix会更新为"llama3_1_8b_rank_0_2"。

策略文件仅在分布式训练任务中保存，用于**权重策略转换**。策略文件以rank_id作为后缀，固定保存为ckpt格式的文件，主要记录了当前rank的网络和优化器切分信息。以单机8卡为例，策略文件保存格式如下：

```text
output/strategy
    ├── ckpt_strategy_rank_0.ckpt
    ├── ckpt_strategy_rank_1.ckpt
    ...
    └── ckpt_strategy_rank_7.ckpt
```

> 注：策略文件保存时会覆盖旧文件，为防止覆盖或混杂不同任务的策略文件，请及时将策略文件保存到自定义文件夹。

可参考[Ckpt权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html)和[Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html)，获取更多权重相关信息。

## YAML参数配置说明

| 参数                     | 描述                                                         |
| ------------------------ | ------------------------------------------------------------ |
| load_checkpoint          | 权重文件或文件夹路径，**断点续训时必填**，默认为空字符串。<br />当配置的路径为空目录时，会退化为使用随机初始化权重进行预训练。<br />若为单卡权重，可配置为权重文件路径，需要确保文件父目录不以"rank_"开头。 |
| src_strategy_path_or_dir | 策略文件或文件夹路径，**`auto_trans_ckpt=True`且load_checkpoint为分布式权重**时需要配置，默认为空字符串。<br />若load_checkpoint配置的权重不带流水线并行切分，则可配置为任一策略文件路径，否则配置为策略文件夹路径。 |
| auto_trans_ckpt          | 权重自动转换开关，load_checkpoint配置的**权重和当前任务的分布式策略不匹配**时需要开启，默认为False。 |
| transform_process_num    | 权重自动转换使用进程数，**仅适用于ckpt格式权重的自动转换**，可加速权重转换。默认为`None`不开启。<br />设置值需要能够整除集群总卡数，设置值越大，host内存占用越高，若host内存不足，需要减少进程数。 |
| resume_training          | 断点续训开关，可设置为`True`或任一rank子文件夹下的权重文件名。默认为`False`。<br />为`True`时，**加载最后保存完整的权重**续训。<br />为权重文件名时，**加载指定step的权重**续训。 |
| load_ckpt_format         | load_checkpoint配置的权重格式，可配置为`safetensors`或`ckpt`，默认为`ckpt`。 |
| remove_redundancy        | 去冗余加载开关，load_checkpoint配置的权重为**去冗余保存的safetensors格式权重**时需要开启，默认为`False`。 |
| load_ckpt_async          | 是否将加载权重与模型编译的操作并行执行。该配置**仅适用于ckpt格式权重且分布式策略不变**的异步加载场景。默认为`False`。 |

## 断点续训使用场景介绍

### 中断续训

**概述**：正常训练任务异常中断，不改变分布式策略，基于保存的权重重新恢复训练任务。

- 基于最后保存完整的权重续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  系统会自动基于各rank的`meta.json`记录的权重，搜索并加载最后保存完整的权重进行续训。

  > 若权重文件夹的所有rank子文件夹下均无meta.json，则退化为基于各自rank最后时间戳的权重续训。

- 基于指定step的权重续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  # 若为ckpt权重，则填写{prefix}-{epoch}_{step}.ckpt
  resume_training: {prefix}-{epoch}_{step}.safetensors
  ```

  用户需确保指定权重的完整性。各rank会自动替换"prefix"中的rank信息来更新要加载的权重名，比如指定的权重名为`llama3_1_8b_rank_0-200_1.safetensors`，rank_1加载时会将权重名替换为`llama3_1_8b_rank_1-200_1.safetensors`。若某rank下权重缺失，会报错权重文件找不到。

### 策略转换续训

**概述**：修改了**分布式策略**或**扩大/缩小集群规模**继续训练任务，需要**开启权重自动转换**。

#### safetensors权重

开启权重自动转换，系统会自动合并safetensors权重为[完整权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html#完整权重)后进行分布式加载，合并的safetensors权重会落盘到`output/unified_checkpoint`文件夹下；若已经将权重离线合并为[完整权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html#完整权重)，则会直接进行分布式加载。离线合并步骤请参考[Safetensors权重-权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html)章节。

- 基于最后保存完整的权重续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  src_strategy_path_or_dir: /path/to/strategy
  resume_training: True
  auto_trans_ckpt: True
  ```

- 基于指定step的权重续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  src_strategy_path_or_dir: /path/to/strategy
  resume_training: {prefix}-{epoch}_{step}.safetensors
  auto_trans_ckpt: True
  ```

- 基于合并的权重续训

  ```yaml
  load_checkpoint: /path/to/unified_checkpoint
  resume_training: True
  auto_trans_ckpt: True
  ```

#### ckpt权重

开启权重自动转换，系统会自动转换权重到当前任务的分布式策略后进行加载，转换的ckpt权重会落盘到`output/transformed_checkpoint`文件夹下，可用于后续直接加载使用且无需开启权重自动转换。

若权重的rank子文件夹下存在多个step的权重文件，需要离线对权重进行筛选，确保**每个rank子文件夹下只有需要加载的单个ckpt文件**。

```yaml
load_checkpoint: /path/to/checkpoint
src_strategy_path_or_dir: /path/to/strategy
resume_training: True
auto_trans_ckpt: True
transform_process_num: 8
```

### 增量续训

**概述**：训练数据集需要**边生产边训练**，当前数据集训练结束后，加入新生产的数据集继续训练，直到所有数据集训练完毕。该场景需要用户基于训练的总数据量，提前预设学习率曲线的总步数。

假设一共训练10T tokens数据，每次生产的数据集只包含1T tokens数据，整个训练过程分10个epoch训完，一共需要花费100000steps。

- 步骤1：预设总训练步数，固定整个训练流程的学习率曲线

  ```yaml
  lr_schedule:
    total_steps: 100000
  ```

- 步骤2：设置足够大的epoch值，确保能够训完所有数据集

  ```yaml
  runner_config:
    epochs: 15
  ```

  > 整个训练过程的学习率曲线已固定，epochs值设置不会影响学习率，可以设置较大值，确保能训完10个数据集。

- 步骤3：数据集训完1个epoch后，可以更换数据集续训，如下为基于最后保存完整的权重续训，其他续训方式请参考[中断续训](#中断续训)或[策略转换续训](#策略转换续训)。

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  > 由于各个数据集样本数量不一致，更换数据集续训，显示的epoch和step可能发生变化，但是当前训练的总step数不变，为正常现象。

### 自动恢复续训

**概述**：为方便平台能够自动拉起断点续训，无需人工干预，可以将load_checkpoint配置为权重checkpoint的保存路径，首次开始训练时，该目录为空，会正常随机初始化权重；续训时，会基于该目录下最后保存完整的权重恢复训练。

```yaml
load_checkpoint: /path/to/output/checkpoint
resume_training: True
```

## 注意事项和建议

- 分布式断点续训必须开启**数据下沉模式**，配置`sink_mode=True`。
- 建议配置`SHARED_PATHS`环境变量为最上层共享目录路径，比如`/data01`是共享目录，工程目录在该目录下，配置`export SHARED_PATHS=/data01`。
- 建议不同分布式策略训练任务的权重和策略文件分开文件夹保存。
