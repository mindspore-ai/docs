# 断点续训

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/resume_training.md)

MindSpore Transformers 动态图（PyNative）支持 **step 级断点续训**：训练中断后，框架从已保存的权重目录恢复模型与优化器状态，将训练进度回拨到中断时的训练步，并将数据集游标跳转到对应位置，从而避免从头重训造成的算力浪费。

续训不需要额外的训练入口，它与首次训练复用同一套 `checkpoint` 配置。差别仅在于**加载侧**的几个开关：将 `load_path` 指向上一次的保存根目录、用 `no_load_optim` 决定是否恢复优化器与训练进度。本页按「机制 → 配置项 → 场景化 YAML」的顺序展开。

> 续训依赖中断前已经将权重和优化器状态落盘。请确认上一次训练满足：
>- `enable_save: True`（默认开启）——否则训练过程中不会产生任何 checkpoint；
>- `no_save_optim: False`（默认）——若保存时跳过了优化器状态，则续训时无法恢复优化器，只能退化为「以权重做初始化」的再训练。

## 一、续训机制

### 1.1 加载入口与触发条件

训练启动时，`Trainer` 会读取 `checkpoint.load_path`（或显式传入的 `checkpoint_path`）作为加载目录；**只要该值非空就会触发加载**。是否恢复优化器与训练进度，则由 `no_load_optim` 进一步决定。

| `no_load_optim` | 加载内容         | 训练进度                               | 数据游标   | 典型用途              |
|-----------------|--------------|------------------------------------|--------|-------------------|
| `False`         | 模型权重 + 优化器状态 | 从 `common.json` 的 `global_step` 恢复 | 跳转到恢复步 | 中断后继续同一次训练        |
| `True`          | 仅模型权重        | 从 0 开始                             | 从头开始   | 以已有权重做初始化的再训练/微调  |

### 1.2 `load_path` 如何定位到具体权重

`load_path` 指向的是**保存根目录**，而非某个具体步的子目录。框架在 `get_checkpoint_path` 中按以下规则定位真正要加载的目录：

1. 校验 `load_path` 存在且是目录；
2. **自训练权重定位**：读取根目录下的 tracker 文件 `latest_checkpointed_iteration.txt`，取出最新的迭代号，拼出 `iteration_XXXXXXXX`（8 位补零）子目录并加载。这意味着续训时**无需手动指定步数，框架会自动选取最近一次保存的 iteration**。

MindSpore Transformers 的训练 checkpoint 默认存储于 `output/checkpoint` 目录，每个 checkpoint 独立保存为以 `iteration` 命名的子文件夹。以 8 卡任务生成的 checkpoint 为例，其保存格式如下：

```text
output
    ├── checkpoint
        ├── latest_checkpointed_iteration.txt               # tracker：记录最新迭代号，假设此时为 2000
        ├── iteration_00001000/                             # 第 1000 步权重目录
            ├── metadata.json                               # 权重切分信息
            ├── common.json                                 # 续训元信息
            ├── {prefix}-model-0000000-0000008.safetensor   # 0 号卡权重分片
            ...
            ├── {prefix}-model-0000007-0000008.safetensor   # 7 号卡权重分片
            ├── {prefix}-opt-0000000-0000008.safetensor     # 0 号卡优化器分片
            ...
            └── {prefix}-opt-0000007-0000008.safetensor     # 7 号卡优化器分片
        ...
        └── iteration_00002000/                             # 第 2000 步权重目录（最新，且所有权重文件都已保存完毕）
```

`load_path` 填写 `output/checkpoint`（根目录）即可，框架会根据 `latest_checkpointed_iteration.txt` 中的内容，自动选择 `iteration_00002000` 下的权重进行加载。

### 1.3 `common.json` 元信息与续训闭环

每个 iteration 子目录都包含一个 `common.json`，记录续训所需的元信息。其配置项由 `CommonInfo` 定义：

| 配置项                 | 含义                        |
|---------------------|---------------------------|
| `global_step`       | 全局训练步（**续训据此恢复进度**）       |
| `global_batch_size` | 保存时的全局批大小（**用于 step 缩放**） |
| `step_num`          | 当前 epoch 内的步号             |
| `epoch_num`         | 已训练的 epoch 数              |
| `loss_scale`        | 梯度缩放系数                    |
| `ckpt_status`       | checkpoint 健康状态标记         |

续训时（`no_load_optim=False`），`Trainer` 会从 `common.json` 读取 `global_step`，将其写入训练状态 `state.global_step`，并调用 `train_dataset.set_init_step(global_step)` 将数据集游标跳到恢复步，确保数据**不重复、不遗漏**。

### 1.4 batch size 变化时的 step 缩放

如果续训时配置的 `global_batch_size` 与保存时不同，直接沿用旧 `global_step` 会导致「已消耗的训练数据量」不一致。为此，`Trainer` 会按照**保存在 `common.json` 中的 `global_batch_size`** 进行等比缩放：

```text
恢复步 = 保存步 × (保存时 global_batch_size / 当前 global_batch_size)
```

> **缩放算例**
>
> 保存时 `global_batch_size=64`、`global_step=1000`（已消耗 64000 个样本）。若续训时改为 `global_batch_size=128`：
>
> ```text
> 恢复步 = 1000 × (64 / 128) = 500
> ```
>
> 即从第 500 步继续，对应的累计数据集消耗量 500 × 128 = 64000，与中断时一致。

### 1.5 仅加载权重时的主权重对齐

当 `no_load_optim: True` 时，不加载优化器状态，但混合精度下的 fp32 主权重仍停留在加载前的初始值。为避免主权重与刚加载的模型参数错位，框架会调用 `optimizer.reload_main_params_from_model()`，用模型参数刷新 fp32 主权重，使两者从同一起点开始。

## 二、续训配置项详解

续训涉及的配置项都在 `checkpoint` 段，由 `CheckpointConfig` 定义。下表先给出速查，随后逐项展开。

| 配置项                      | 默认值      | 作用侧 | 说明                          |
|--------------------------|----------|-----|-----------------------------|
| `enable_save`            | `True`   | 保存  | 是否保存 checkpoint             |
| `save_path`              | `""`     | 保存  | 权重保存在此目录下                   |
| `save_interleaved_steps` | `1000`   | 保存  | 每间隔多少步保存一次                  |
| `no_save_optim`          | `False`  | 保存  | 是否跳过保存优化器状态                 |
| `save_max`               | `5`      | 保存  | 当前任务开始后，最多新保存多少份 checkpoint |
| `load_path`              | `""`     | 加载  | 续训加载的保存根目录                  |
| `no_load_optim`          | `False`  | 加载  | 是否跳过加载优化器状态（决定续训/仅权重初始化）    |
| `load_balanced`          | `False`  | 加载  | 是否启用均衡加载（与并行配合使用，见 2.3）     |

### 2.1 保存侧配置项（决定能否续训）

当前任务进行续训时，需要确保这些配置在上次训练时开启，而且权重被完整地保存到磁盘中。下表列出保存侧各配置项的参数说明。

| 参数名称                     | 数据类型 | 是否可选 | 默认值     | 取值说明                                                                                                  |
|--------------------------|------|------|---------|-------------------------------------------------------------------------------------------------------|
| `enable_save`            | bool | 可选   | `True`  | 是否开启权重保存。关闭后整个训练过程不落盘。仅在调试、不需要中断恢复时关闭。                                                                |
| `save_path`              | str  | 必选   | `""`    | 保存根目录，每次保存会在其下生成 `iteration_XXXXXXXX` 子目录与 tracker 文件。续训时一般把 `load_path` 设为同一目录。                      |
| `save_interleaved_steps` | int  | 可选   | `1000`  | 保存间隔步数。仅当 `global_step % save_interleaved_steps == 0` 时才会落盘。步数越小越能抵抗中断风险，但 IO 开销越大；大规模预训练通常设置为几百到几千步。 |
| `no_save_optim`          | bool | 可选   | `False` | 保存时是否跳过优化器状态。`True` 时只存模型权重。默认 `False`，保证后续可以恢复优化器、完整续训。仅在确定不再需要继续优化（如只想导出权重）时设为 `True`。              |
| `save_max`               | int  | 可选   | `5`     | 当前训练任务最多保留的 checkpoint 份数，超出后自动清理最旧的。按磁盘容量与回退需求设置。                                                    |

### 2.2 加载侧配置项（决定续训行为）

下表列出加载侧各配置项的参数说明。

| 参数名称             | 数据类型  | 是否可选 | 默认值      | 取值说明                                                                                                                                                                                                                                               |
|------------------|-------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load_path`      | str   | 必选   | `""`     | 续训加载的保存根目录。非空即触发加载，框架会自动通过 tracker 选择最新 iteration（见 1.2）。指向上一次训练的 `save_path`（自训练权重目录，需包含 `common.json`/`metadata.json`）。                                                                                                                          |
| `no_load_optim`  | bool  | 可选   | `False`  | 加载时是否跳过优化器状态。它是「续训」与「仅权重初始化」的总开关（见 1.1）。续训中断的训练设为 `False`（恢复优化器 + 步数 + 数据游标）；以已有权重重新训练/微调设为 `True`（只取权重作初始化）。`no_load_optim: False` 时优化器对象必须存在，否则会报错。                                                                                              |
| `load_balanced`  | bool  | 可选   | `False`  | 开启后，加载阶段会先使用 shard 均衡策略（`apply_balance_shard_strategy`）计算各 rank 间的冗余参数映射，再通过 `single_parameter_broadcast` 将参数在 rank 之间广播，从而消除冗余参数的重复加载，降低分布式加载时的显存与 IO 开销。仅在分布式并行训练中有意义。单卡或非并行场景下无收益，保持默认 `False` 即可。并行维度的含义与配置见[分布式并行训练](./parallel_training.md)。 |

## 三、续训与加载权重训练的区别

下表对比了加载权重训练与断点续训的配置差异，并补充了**默认值**，帮助区分「配置项默认值」与「实际应当如何配置」：

| 项目               | 加载权重训练                            | 断点续训                               | 默认值                              |
|------------------|-----------------------------------|------------------------------------|----------------------------------|
| `load_path`      | 为空（随机初始化）或指向预训练权重目录               | 指向上一次的保存根目录                        | 默认值： `""`；非空即加载。                 |
| `no_load_optim`  | 通常显式置 `True`（加载权重进行预训练时，不需要优化器信息） | 必须为 `False`（恢复优化器）                 | **默认值： `False`**                 |
| 起始 `global_step` | 从 0 开始                            | 从 `common.json` 的 `global_step` 恢复 | 默认值： `0`                         |
| 数据集游标            | 从头开始                              | 跳转到恢复步                             | `set_init_step`                  |
| fp32 主权重         | 随网络初始化                            | `no_load_optim=True` 时从模型参数中获取     | `reload_main_params_from_model`  |

> **不要被 no_load_optim 的默认值误导**
>
> `no_load_optim` 的源码默认值是 `False`。这对「续训」恰好正确（默认就会恢复优化器），但对「首次预训练」并不合适——首次训练通常根本没有可加载的优化器状态，常见做法是**不设 `load_path`**（不加载），或在以预训练权重做初始化时**显式将 `no_load_optim` 设为 `True`**。
>
> 即：`no_load_optim` 的默认值会默认走入续训流程，若为加载权重训练，则需要对该配置按场景进行调整。

## 四、场景化配置示例

以下按三类典型场景给出可直接使用的完整 `checkpoint` 段 YAML。其余段（模型、数据集、并行等）参见[配置文件说明](./configuration.md)。

### 场景 A：标准续训（恢复优化器 + 步数 + 数据游标）

最常见的场景：训练意外中断，从最近一次保存继续。`no_load_optim: False`，框架自动选最新 iteration、恢复 `global_step` 并跳转数据游标。

```yaml
checkpoint:
  # 保存配置（沿用上一次训练，确保中断前已落盘）
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  no_save_optim: False
  save_max: 5

  # 续训加载配置
  load_path: "./output/ckpt"   # 指向保存根目录，框架自动取最新 iteration
  no_load_optim: False         # 恢复优化器状态、训练步与数据游标
  load_balanced: False         # 单机/非均衡场景保持 False
```

### 场景 B：仅以权重做初始化的再训练 / 微调

只想复用已有权重作为起点（例如使用预训练权重进行微调），不沿用旧的优化器状态与训练进度。设置 `no_load_optim: True`，训练将从第 0 步重新开始，并触发 fp32 主权重对齐。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/finetune_ckpt"
  save_interleaved_steps: 500
  no_save_optim: False
  save_max: 5

  # 仅加载权重作为初始化
  load_path: "./pretrained/ckpt"  # 自训练权重根目录（具体 iteration 下的文件包含 common.json/metadata.json）
  no_load_optim: True             # 不恢复优化器/训练步数/数据游标，从 step 0 开始
  load_balanced: False
```

> 当前动态图不支持直接加载 HuggingFace 权重目录。

### 场景 C：改 `global_batch_size` 续训

续训时调整了全局批大小（例如扩展或缩减节点数量）。配置保持不变，框架会按照 1.4 节的公式自动缩放恢复步，无需手动修改 `global_step`。

```yaml
# training 段：续训时把 global_batch_size 从 64 改为 128
training:
  global_batch_size: 128

checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  no_save_optim: False
  save_max: 5

  load_path: "./output/ckpt"
  no_load_optim: False    # 必须为 False，缩放逻辑只在恢复优化器时生效
  load_balanced: False
```

按算例，若保存时 `global_batch_size=64`、`global_step=1000`，续训改为 `128` 后将从第 `1000 × 64 / 128 = 500` 步继续，累计数据集消耗量保持 64000 不变。

> 缩放仅在续训路径生效。
> step 缩放仅发生在 `no_load_optim: False` 时。若配置 `no_load_optim: True`（即不加载优化器权重），训练则从 0 开始，不进行数据集缩放。

### 场景 D：分布式并行下的均衡加载

使用多卡进行分布式并行续训时，开启 `load_balanced` 可以消除冗余参数的重复加载，降低加载时的显存占用与 IO 开销。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  no_save_optim: False
  save_max: 5

  load_path: "./output/ckpt"
  no_load_optim: False
  load_balanced: True      # 仅并行场景有意义：shard 均衡 + 参数广播
```

并行维度的设置见[分布式并行训练](./parallel_training.md)。

## 相关文档

- 权重格式与保存机制：[权重保存与加载](./save_load_checkpoint.md)
- 配置项总览：[配置文件说明](./configuration.md)
- `load_balanced` 与并行维度：[分布式并行训练](./parallel_training.md)
- 数据游标恢复与 `set_init_step`：[数据集](./dataset.md)
- 端到端训练流程：[训练指南](../guide/training.md)
- 快速上手：[快速开始](../quick_start/quick_start.md)
