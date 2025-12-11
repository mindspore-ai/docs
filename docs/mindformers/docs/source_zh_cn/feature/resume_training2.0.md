# 断点续训2.0

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/resume_training2.0.md)

## 概述

MindSpore Transformers 具备完备的断点续训能力，核心功能与适用场景如下：

1. **核心功能**：支持加载已保存的checkpoint，快速恢复训练进度，无需从零开始；
2. **多场景适配**：覆盖四大主流续训场景
   - **中断续训**：正常训练任务异常中断（如设备故障、网络波动）后，基于已保存的checkpoint重新恢复训练流程；
   - **扩缩容续训**：训练过程中调整卡数（扩容 / 缩容），基于已保存的checkpoint继续训练；
   - **增量续训**：在已有训练成果基础上，补充训练数据集，基于已保存的checkpoint继续训练；
   - **自动恢复续训**：支持平台无需人工干预自动拉起断点续训；

对于大规模训练任务（训练周期长、资源投入大），可避免意外中断导致的进度丢失，显著减少时间与计算资源浪费。

> 本文档仅适用于使用 [Checkpoint 2.0 版本](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/checkpoint_saving_and_loading.html)进行续训的场景；若用户使用Checkpoint 1.0 版本，需参考旧版[断点续训文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/resume_training.html)。

## checkpoint介绍

MindSpore Transformers 的训练checkpoint默认存储于 `output/checkpoint` 目录，每个checkpoint独立保存为以 `iteration` 命名的子文件夹。以 8 卡任务第 1 步生成的checkpoint为例，其保存格式如下：

```text
output
    ├── checkpoint
        ├── iteration_0000001
            ├── metadata.json
            ├── common.json
            ├── {prefix}-model-0000000-0000008.safetensor
            ...
            ├── {prefix}-model-0000007-0000008.safetensor
            ├── {prefix}-opt-0000000-0000008.safetensor
            ...
            └── {prefix}-opt-0000007-0000008.safetensor
        ...
        └── latest_checkpointed_iteration.txt
```

可参考[checkpoint保存和加载](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/checkpoint_saving_and_loading.html)，获取更多checkpoint相关信息。

## 配置说明

| 参数名称        | 描述                                                         | 取值说明                       |
| --------------- | ------------------------------------------------------------ | ------------------------------ |
| load_checkpoint | checkpoint文件夹路径，可**填写`output/checkpoint`文件夹路径或`iteration`子文件夹路径**。<br />若为`checkpoint`文件夹路径，将会按照`latest_checkpointed_iteration.txt`中记录的迭代步数，加载对应`iteration`子文件夹checkpoint。 | (str，可选) - 默认值：`""`     |
| resume_training | 断点续训功能开关，设置为 `True` 时，将从待加载checkpoint对应的迭代步数继续训练。 | (bool，可选) - 默认值：`False` |

## 场景介绍

### 中断续训

**概述**：正常训练任务异常中断后，在不改变分布式策略的前提下，基于已保存的checkpoint重新恢复训练流程。

MindSpore Transformers 支持用户使用以下两种方式启动断点续训：

- 基于`latest_checkpointed_iteration.txt`中记录的迭代步数续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

- 基于指定迭代步数续训

  ```yaml
  load_checkpoint: /path/to/checkpoint/iteration_{x}
  resume_training: True
  ```

  > x 代表checkpoint对应的训练迭代步数，例如 "0000001" 即表示第 1 步训练对应的checkpoint。

### 扩缩容续训

**概述**：需要**扩大/缩小集群规模**或**修改分布式策略**继续训练任务，配置方式和[中断续训](#中断续训)一致。MindSpore Transformers 依托在线 Reshard 机制，可确保checkpoint权重自动适配任意分布式策略，保障续训顺畅。

- 基于`latest_checkpointed_iteration.txt`中记录的迭代步数续训

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

- 基于指定迭代步数续训

  ```yaml
  load_checkpoint: /path/to/checkpoint/iteration_{x}
  resume_training: True
  ```

  > x 代表checkpoint对应的训练迭代步数，例如 "0000001" 即表示第 1 步训练对应的checkpoint。

### 增量续训

**概述**：训练数据集需要**边生产边训练**，当前数据集训练结束后，加入新生产的数据集继续训练，直到所有数据集训练完毕。该场景需要用户根据训练的总数据量，提前预设学习率曲线的总步数。

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

- 步骤3：数据集训完1个epoch后，可以更换数据集续训，如下为基于`latest_checkpointed_iteration.txt`中记录的迭代步数续训，其他续训方式请参考[中断续训](#中断续训)或[扩缩容续训](#扩缩容续训)。

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  > 更换数据集续训时，因各数据集样本数量不同，显示的 epoch 和单批次 step 可能变化，但训练总 step 数保持不变，这属于正常现象。

### 自动恢复续训

**概述**：为支持平台无人工干预自动拉起断点续训，可将 `load_checkpoint` 配置为checkpoint保存目录路径：首次训练时目录为空，模型随机初始化参数；续训时则基于该目录下最后保存的完整checkpoint恢复训练。

```yaml
load_checkpoint: /path/to/output/checkpoint
resume_training: True
```

## 约束说明

- 多机场景下，断点续训需将所有checkpoint文件存放于同一共享目录，用户需将该共享路径配置至环境变量 `SHARED_PATHS`；建议优先配置最上层共享目录，示例：共享目录为 `/data01` 时，执行 `export SHARED_PATHS=/data01` 即可。
