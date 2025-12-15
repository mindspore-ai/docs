# checkpoint保存和加载

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/checkpoint_saving_and_loading.md)

## 概述

MindSpore Transformers 支持训练过程中保存checkpoint。checkpoint包括**模型权重**、**优化器权重**、**训练上下文信息**和**分布式策略元信息**，核心作用是**中断后恢复训练**、**防止训练失败丢失进度**，同时支持**后续微调**、**推理**或**模型迭代**。

MindSpore Transformers 推出**Checkpoint 2.0 版本**，通过重构checkpoint保存策略与加载流程，实现易用性与加载效率的综合提升。

相较于Checkpoint 1.0 版本，核心更新如下：

- **全新checkpoint保存[目录结构](#目录结构)**：目录包含**模型权重**、**优化器权重**、**训练上下文信息**、**分布式策略元信息**等文件；
- **新增在线 Reshard 加载机制**：若待加载checkpoint的分布式策略元信息与当前任务不一致，加载时将**自动对权重参数执行 Reshard 转换**，生成适配当前分布式策略的参数；
- **简化加载配置**：依托在线 Reshard 机制，用户**无需手动配置`auto_trans_ckpt`、`src_strategy_path_or_dir`等参数**触发权重策略转换，易用性显著提升。

MindSpore Transformers 目前默认采用Checkpoint 1.0 版本，用户需在 YAML 配置文件中添加如下参数，即可启用Checkpoint 2.0 版本的保存与加载功能。

```yaml
use_legacy_format: False
```

> 该文档仅针对用户使用体验Checkpoint 2.0 版本，若使用Checkpoint 1.0 版本，请参考[Safetensors文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html)或[Ckpt文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html)。

## checkpoint保存

### 目录结构

MindSpore Transformers 的训练checkpoint默认存储于 `output/checkpoint` 目录，每个checkpoint独立保存为以 `iteration` 命名的子文件夹。以 8 卡任务第 1 步生成的checkpoint为例，其保存格式如下：

```text
output
    ├── checkpoint
        ├── iteration_00000001
            ├── metadata.json
            ├── common.json
            ├── {prefix}-model-0000000-0000008.safetensors
            ...
            ├── {prefix}-model-0000007-0000008.safetensors
            ├── {prefix}-opt-0000000-0000008.safetensors
            ...
            └── {prefix}-opt-0000007-0000008.safetensors
        ...
        └── latest_checkpointed_iteration.txt
```

权重相关文件说明

| 文件                                       | 描述                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| metadata.json                              | 记录各参数的分布式策略元信息与存储信息，为后续加载权重时自动执行 Reshard 转换提供必要的元数据支持，确保转换精准适配当前任务。 |
| common.json                                | 记录当前迭代（iteration）的训练信息，为断点续训提供数据支持。 |
| {prefix}-model-0000000-0000008.safetensors | 模型权重存储文件。命名规则说明：`prefix` 为自定义文件名前缀，`model` 标识文件类型为模型权重，`0000000` 是文件序号，`0000008` 代表总文件个数。 |
| {prefix}-opt-0000000-0000008.safetensors   | 优化器权重存储文件。命名规则说明：`prefix` 为自定义文件名前缀，`opt` 标识文件类型为优化器权重，`0000000` 是文件序号，`0000008` 代表总文件个数。 |
| latest_checkpointed_iteration.txt          | 记录 `output/checkpoint` 目录下最后一个成功保存的checkpoint对应的迭代步数。 |

### 配置说明

用户可通过修改 YAML 配置文件中 `CheckpointMonitor` 下的相关字段，控制权重保存行为，具体参数说明如下：

| 参数名称              | 描述                                                         | 取值说明                                                     |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| prefix                | 权重文件名自定义前缀，建议填写模型名称以区分不同模型的checkpoint。 | (str, 可选) - 默认值： `"CKP"` 。                            |
| directory             | checkpoint保存路径，未配置时默认存储于 `./output/checkpoint`。 | (str, 可选) - 默认值： `None` 。                             |
| save_checkpoint_steps | 设置保存checkpoint的训练间隔步数（即每训练指定步数保存一次checkpoint）。 | (int, 可选) - 默认值： `1` ，不设置时不保存模型权重。        |
| keep_checkpoint_max   | 设置checkpoint最大保留数量，达到上限后，保存新checkpoint时会自动删除最旧的checkpoint。 | (int, 可选) - 默认值： `5` 。                                |
| async_save            | checkpoint异步保存功能开关（控制是否启用异步保存机制）。     | (bool, 可选) - `True` 时将使用异步线程保存checkpoint。默认值： `False` 。 |
| checkpoint_format     | checkpoint权重保存格式，Checkpoint 2.0 版本仅支持 `'safetensors'`；若已配置 `use_legacy_format: False`，该字段将自动转换为 `'safetensors'`。 | (str, 可选) - 默认值： `'safetensors'` 。                    |
| remove_redundancy     | checkpoint去冗余保存功能开关（控制是否启用去冗余保存机制）。 | (bool, 可选) - 默认值： `False` 。                           |
| save_optimizer        | 优化器权重保存功能开关（控制是否保存优化器权重信息）。       | (bool, 可选) - 默认值： `True` 。                            |

配置示例如下：

```yaml
callbacks:
  ...
  - type: CheckpointMonitor
    prefix: "qwen3"
    save_checkpoint_steps: 1000
    keep_checkpoint_max: 5
    async_save: False
    checkpoint_format: "safetensors"
    save_optimizer: True
  ...
```

> 上述配置指定训练任务以 "qwen3" 作为 safetensors 文件名前缀，采用同步保存模式，每 1000 步保存一次包含模型权重与优化器权重的checkpoint，且训练全程最多保留最新的 5 个checkpoint。

如果您想了解更多有关 CheckpointMonitor 的知识，可以参考 [CheckpointMonitor API 文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/core/mindformers.core.CheckpointMonitor.html)。

## checkpoint加载

MindSpore Transformers 提供灵活的checkpoint加载能力，覆盖单卡与多卡全场景，核心特性如下：

1. Checkpoint 2.0 版本适配性升级：依托在线 Reshard 机制，加载时权重可自动适配任意分布式策略任务，无需手动调整，降低多场景部署成本；
2. 跨平台权重兼容：通过专用转换接口，支持加载 HuggingFace 社区发布的权重文件，当前已实现 Qwen3 模型训练场景的兼容适配，方便用户复用社区资源。

### 配置说明

用户可通过修改 YAML 配置文件中的相关字段，控制权重加载行为。

| 参数名称             | 描述                                                         | 取值说明                       |
| -------------------- | ------------------------------------------------------------ | ------------------------------ |
| load_checkpoint      | checkpoint文件夹路径，可**填写`output/checkpoint`文件夹路径或`iteration`子文件夹路径**。<br />若为`checkpoint`文件夹路径，按照`latest_checkpointed_iteration.txt`中记录的步数加载对应`iteration`子文件夹checkpoint。 | (str，可选) - 默认值：`""`     |
| pretrained_model_dir | 指定 HuggingFace 社区权重的文件夹路径；若同时配置了 `load_checkpoint`，该字段将自动失效。 | (str，可选) - 默认值：`""`     |
| balanced_load        | 权重均衡加载功能开关，**仅支持在分布式任务中开启**；设为 `True` 时，各 rank 按参数均衡分配策略加载权重，再通过参数广播获取最终权重。 | (bool，可选) - 默认值：`False` |
| use_legacy_format    | Checkpoint 1.0 版本启用开关，需设置为 `False`（使用Checkpoint 2.0 版本）。 | (bool，可选) - 默认值：`True`  |
| load_ckpt_format     | 指定加载权重的格式，需设置为 `'safetensors'`（适配Checkpoint 2.0 版本）。 | (str，可选) - 默认值：`'ckpt'` |

当 `load_checkpoint` 配置为 `output/checkpoint` 文件夹路径时，用户可通过修改 `latest_checkpointed_iteration.txt` 中记录的步数，实现指定 `iteration` 权重的加载。

## 约束说明

- 多机场景下，所有文件需存储于**同一共享目录**，用户需将该**共享路径配置至环境变量 `SHARED_PATHS`**。建议优先配置为最上层共享目录路径，示例：若共享目录为 `/data01`（工程目录位于其下），可执行 `export SHARED_PATHS=/data01`。
