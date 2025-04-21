# 权重保存与断点续训

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/function/resume_training.md)

## 权重保存

### 概述

在深度学习模型的训练过程中，保存模型的权重是至关重要的一步。权重保存功能使得我们能够在训练的任意阶段存储模型的参数，以便用户在训练中断或完成后进行恢复、继续训练、评估或部署。通过保存权重。同时还可以在不同环境下复现实验结果。

### 目录结构

在训练过程中，MindSpore Transformers会在输出目录中生成两个权重保存文件夹：`checkpoint` 和 `checkpoint_network`。

| 文件夹                | 描述                                                  |
|--------------------|-----------------------------------------------------|
| checkpoint         | 保存权重、优化器状态、step和epoch于ckpt文件中，用于**断点恢复训练**。         |
| checkpoint_network | 仅保存权重参数于ckpt文件中，适用于作为**预训练权重**的加载或**推理评估**，不支持断点续训。 |

#### `checkpoint`目录结构

`checkpoint`文件夹中的权重文件按如下格式保存：

```text
checkpoint
  ├── rank_0
    ├── meta.json
    └── {prefix}-{epoch}_{step}.ckpt
  ...
  └── rank_x
    ├── meta.json
    └── {prefix}-{epoch}_{step}.ckpt
```

| 文件                           | 描述                                                                                                                                                                                                                                                 |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| meta.json                    | 记录最后保存的权重的`epoch`、`step`和权重名，每个rank进程独立维护一个`meta.json`文件。                                                                                                                                                                                          |
| {prefix}-{epoch}_{step}.ckpt | 保存的权重文件，`prefix`包含rank_id信息，格式为`{prefix}-{epoch}_{step}.ckpt`。如果前缀相同的文件已经存在，系统会自动递增后缀。开启数据下沉时，`epoch`位置计算方式为 $\frac{CurrentTotalStepNumber}{SinkSize} = \frac{((CurrentEpoch-1)*StepsPerEpoch+CurrentStepInEpoch)}{SinkSize}$，`step`固定为`sink_size` |

#### `checkpoint_network`目录结构

```text
checkpoint
  ├── rank_0
    └── {prefix}-{epoch}_{step}.ckpt
  ...
  └── rank_x
    └── {prefix}-{epoch}_{step}.ckpt
```

| 文件                           | 描述                                                                                                    |
|------------------------------|-------------------------------------------------------------------------------------------------------|
| {prefix}-{epoch}_{step}.ckpt | 保存的权重文件，`prefix`包含rank_id信息，格式为`{prefix}-{epoch}_{step}.ckpt`。如果前缀相同的文件已经存在，系统会自动递增后缀。开启数据下沉时的命名规则同上。 |

### 配置与使用

#### YAML参数配置

用户可通过修改配置文件来控制权重保存的行为。以下是主要参数：

| 参数                    | 描述                                |
|-----------------------|-----------------------------------|
| save_checkpoint_steps | 每多少步保存一次权重，不设置时为不保存。              |
| keep_checkpoint_max   | 最多同时保存多少个权重文件，达到上限后会在保存权重时删除最旧的权重文件。 |

用户可修改`yaml`配置文件中`CheckpointMonitor`下的字段来控制权重保存行为。例如：

```yaml
callbacks:
  ...
  - type: CheckpointMonitor
    prefix: "llama2_7b"
    save_checkpoint_steps: 500
    keep_checkpoint_max: 3
  ...
```

上例中表示每隔500步保存一次权重，最多同时存储三个权重。

## 断点续训

### 概述

MindSpore Transformers支持**step级断点续训**功能，允许在训练中保存模型的checkpoint，并在训练中断后，加载保存的checkpoint恢复之前的状态继续训练。这一特性在处理大规模训练任务时尤为重要，能够有效减少因意外中断导致的时间和资源浪费。此外，在数据集不变，但`global batch size`改变的断点续训场景下，例如更换集群或修改配置时，本工具还支持续训步数与数据跳过步数自动同比例缩放。

### 配置与使用

#### YAML参数配置

用户可通过修改配置文件来控制断点续训的行为。以下是主要参数，其他参数可参考CheckpointMonitor介绍：

| 参数            | 描述                                                                                                           |
| --------------- |--------------------------------------------------------------------------------------------------------------|
| load_checkpoint | 断点续训时加载的权重路径。路径可以是文件夹路径（用于加载分布式权重），也可以是具体权重文件的路径。默认为空字符串，即不加载权重（断点续训时必填）                                     |
| resume_training | 断点续训开关，可设置为`True`或指定特定的权重文件名。为`True`时，系统会自动从上次中断处恢复训练。默认为`False`                                             |
| load_ckpt_async | 是否将加载权重与模型编译的操作并行执行，不支持在线自动切分权重场景（auto_trans_ckpt=True），该场景下不生效。默认为False串行执行。<br />为`True`时，并行执行，减少总体拉起续训的耗时 |

根据传入参数不同，可分为如下四种情况：

| load_checkpoint | resume_training | 功能描述                                                                                                                                                                    | 是否为推荐使用方式 |
|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| 权重文件路径          | True            | 基于load_checkpoint指代的权重续训                                                                                                                                                | √         |
| 权重文件路径          | 权重文件名           | resume_training指代的文件名无效，基于load_checkpoint指代的权重续训                                                                                                                        | ×         |
| 权重文件夹路径         | True            | **场景1："单机"或"多机+共享目录"或"ModelArts"**<br />① 基于meta.json记录的权重续训，支持故障恢复。<br />② 若任一rank文件夹下缺少meta.json，所有rank基于最后时间戳的权重续训。<br />**场景2："多机+非共享目录"**<br />所有rank基于最后时间戳的权重续训。 | √         |
| 权重文件夹路径         | 权重文件名           | 基于resume_training指代的权重续训                                                                                                                                                | √         |

此外，用户还可通过增改配置文件的如下参数来使用相关功能。

| 参数               | 描述                                                                                                          |
|------------------|-------------------------------------------------------------------------------------------------------------|
| ignore_data_skip | 是否忽略断点续训时跳过数据的机制，而从头开始读取数据集。用于续训时数据集更换的场景。设置为`True`时不会跳过数据集，默认为`False`。                                     |
| data_skip_steps  | 数据集跳过步数。用于更换数据集续训后再次断开续训或`global batch size`改变的场景，须手动设置此参数来配置新数据集跳过步数，如`global batch size`改变，需向下整除缩放系数后再传入。 |

#### 故障恢复机制

当`resume_training`设置为`True`时，系统会自动基于`meta.json`记录的权重进行续训。如果某个rank的权重文件缺失或损坏，系统会回退到上一个可用的权重进行恢复。

> 分布式环境中，断点续训要求所有节点的权重在同一共享目录下。用户可通过环境变量`SHARED_PATHS`来设置共享路径。

### 分布式训练示例

以下示例演示了如何在单卡和多卡环境中启动断点续训。示例基于`llama2_7b`
模型，相关配置文件[configs/llama2/pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml)。

#### 完整训练

1. 修改`configs/llama2/pretrain_llama2_7b.yaml`：

   根据需要设置并行配置：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 2
     pipeline_stage: 2
     micro_batch_num: 2
   ```

   根据需要设置模型权重保存配置：

   ```yaml
   callbacks:
     ...
     - type: CheckpointMonitor
       prefix: "llama2_7b"
       save_checkpoint_steps: 10
       keep_checkpoint_max: 3
       integrated_save: False
       async_save: False
     ...
   ```

2. 准备数据集，此处以[wikitext2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)为例，启动4卡分布式训练：

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   在第四次保存完毕后，结束进程，此时`checkpoint`下的`rank_0`文件夹结构为：

   ```text
   checkpoint/rank_0
     ├── llama2_7b_rank_0-10_2.ckpt
     ├── llama2_7b_rank_0-15_2.ckpt
     ├── llama2_7b_rank_0-20_2.ckpt
     └── meta.json
   ```

#### 断点续训

1. 修改配置，指定断点续训权重文件：

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ```

2. 启动断点续训：

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   如若初始步数从第`42`步开始，则断点续训成功。由于最后保存的权重包含了第`40`步的信息，`sink_size`默认为`2`，即每两步打印一次信息，因此初始步数为`42`。

#### 切换数据集断点续训

在切换数据集并进行断点续训时，有三种主要场景，每个场景需要针对配置文件进行不同的修改。下面逐一介绍每种情况，并详细说明在哪些场景下需要对基本断点续训流程的哪一步进行修改，以及如何修改具体配置来达成预期效果。

**场景一：全新数据集，继续训练（无需跳过已训练的步数）**

在这种场景中，当切换到一个全新数据集时，模型的训练将从新数据集的开头开始，而无需跳过任何步数。对于这种情况，配置文件需要设置为**忽略之前的数据进度**，让模型在新数据集上从头训练。

- **配置修改**：需要在基本断点续训流程的第一步的基础上对`ignore_data_skip`进行设置。将`ignore_data_skip`设置为`True`，表示不跳过数据集。

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: True
   ```

- **预期效果**：模型将在新数据集上从头训练，而不会跳过任何步数。

**场景二：在新数据集上断点续训，并跳过部分已训练的步数**

在这种情况下，模型在新数据集上已经训练了一部分（例如断开前已训练了`2`步），期望从上次中断的地方继续训练。此时，必须手动指定需要跳过的步数。

- **配置修改**：需要在基本断点续训流程的第一步的基础上对`ignore_data_skip`和`data_skip_steps`进行设置。将`ignore_data_skip`设置为`False`，并且通过`data_skip_steps`指定要跳过的已训练步数（例如，跳过`2`步）。

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: False
   data_skip_steps: 2
   ```

- **预期效果**：模型将跳过新数据集的前`2`步，从第`3`步开始继续训练。

**场景三：在新数据集上断点续训时，`global batch size`发生变化**

如果在新数据集上继续断点续训时，`global batch size`改变了（例如，变为原先的 2 倍），手动指定需跳过的步数时需要对已训练的步数进行缩放。具体来说，跳过的步数需要根据缩放系数向下整除。例如，如果`global batch size`变为原先的`2`倍，需跳过的步数则相应减少一半。

- **配置修改**：需要在场景二的基础上对`data_skip_steps`进行调整。将`data_skip_steps`设置为缩放后的步数。例如，`global batch size`变为原先的`2`倍，需跳过的步数变为`1`（向下整除）。

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: False
   data_skip_steps: 1
   ```

- **预期效果**：模型将根据新的`global batch size`调整跳过的步数，并从正确的地方继续训练。

#### 故障恢复示例

当部分权重文件缺失时，系统会自动基于上一个可用的权重进行恢复。

1. 删除`rank_3`下的`llama2_7b_rank_0-20_2.ckpt`文件。删除后文件夹结构应为：

   ```text
   checkpoint/rank_3
     ├── llama2_7b_rank_0-10_2.ckpt
     ├── llama2_7b_rank_0-15_2.ckpt
     └── meta.json
   ```

2. 修改配置，启用故障恢复：

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ```

3. 启动分布式训练：

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   如若初始步数从第`32`步开始，则断点续训成功。由于`rank_3`下的包含了第`40`步的信息的权重被删除，因此自动使用上一次保存的权重，即包含第
   `30`步信息的权重。由于`sink_size`默认为`2`，即每两步打印一次信息，因此初始步数为`32`。

### 注意事项

- **数据下沉模式**：分布式断点续训必须开启数据下沉模式，配置`sink_mode=True`。
- **权重文件检查**：确保断点续训加载的权重为训练中断时的权重，而不是整个训练过程最后保存的权重，否则会报错。
