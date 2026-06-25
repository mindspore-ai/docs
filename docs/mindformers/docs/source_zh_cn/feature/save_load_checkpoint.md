# 权重保存与加载

MindSpore Transformers 动态图（PyNative）训练统一以 **Safetensors** 格式保存与加载权重。框架在 `checkpoint` 段（`CheckpointConfig`）集中配置保存与加载行为：保存由 [mindformers/pynative/callback/checkpoint_callback.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/callback/checkpoint_callback.py) 的 `CheckpointCallback` 在训练过程中按步触发，加载由 `Trainer` 在训练启动阶段完成。两者底层分别调用 `mindformers.checkpoint.save_checkpoint` 与 `mindformers.checkpoint.load_checkpoint`，并通过权重目录内的 `common.json` 衔接断点续训。

> **动态图仅用 Safetensors**
>
> 动态图路径全程以 Safetensors 为唯一权重格式，**不涉及** ckpt 与 Safetensors 之间的格式转换，也不需要单独的转换脚本。

## 术语说明

本文涉及以下术语：

- **权重加载（Weight Loading）**

  仅加载模型参数，不恢复优化器状态和训练进度。常用于微调、蒸馏、迁移学习等场景。

- **断点续训**

  与整段续训含义相同，强调从某个 checkpoint 恢复训练过程。

- **Checkpoint**

  训练过程中保存的模型状态文件，用于恢复训练或导出模型。

- **Global Step**

  训练过程中已经执行的优化步数，用于学习率调度、训练恢复以及训练统计。

## 如何选择 Checkpoint 配置

请根据实际需求选择保存和加载方式：

```text
是否需要从中断位置继续训练？
│
├── 是
│   └── 使用整段续训
│       ├── 保存 optimizer 状态
│       ├── 保存训练进度
│       └── 加载时使用 resume 配置
│
└── 否
    │
    ├── 仅用于模型推理
    │   └── 仅保存权重
    │
    └── 用于微调新任务
        └── 加载模型权重，不加载优化器状态
```

常见场景对应关系：

| 场景     | 推荐方式 |
| ------ | ---- |
| 训练中断恢复 | 整段续训 |
| 集群故障恢复 | 整段续训 |
| 模型微调   | 权重加载 |
| 模型推理部署 | 权重加载 |
| 模型转换导出 | 权重加载 |

## 按场景选哪些字段

下表帮助快速定位每个场景需要关注的字段，详细语义见后文对应小节。

| 场景                 | 关键字段 | 概述                                    |
|--------------------|---|---------------------------------------|
| 关闭保存（只显示 loss 不保存权重） | `enable_save: False` | 不挂保存回调，仅保留 Loss/Monitor               |
| 基础定时保存             | `save_path` / `save_interleaved_steps` / `save_max` | 按步存、保留最近若干份                           |
| 异步保存               | `async_save: True` | 落盘与计算重叠，降低保存阻塞                        |
| 仅存权重（不保存优化器权重）     | `no_save_optim: True` | 体积更小，但无法保存优化器状态                       |
| 去冗余保存              | `remove_redundancy: True` | 多卡分片去重，减小占用                           |
| 多卡布局缓存             | `save_global_layout_cache` | 复用分片元信息，避免每次重算                        |
| 全量断点续训             | `load_path` + `no_load_optim: False` | 恢复权重、优化器、step 与数据游标                   |
| 微调仅加载权重            | `load_path` + `no_load_optim: True` | 只取权重，优化器从头开始                          |
| 多卡均衡加载             | `load_balanced: True` | shard 均衡 + 参数广播，消除冗余参数重复加载，仅多卡分片场景有意义 |

> **字段归属**
>
> 保存字段被 `CheckpointCallback` 消费，加载字段被 `Trainer._load_checkpoint` 消费。两类字段都写在同一个 `checkpoint` 段下，互不影响。

---

## 保存

### 触发机制

保存逻辑全部在 `CheckpointCallback` 中，关键行为如下：

- **是否挂载回调**：`Trainer._create_built_in_callbacks` 检查 `enable_save`。当 `enable_save=False` 时**根本不会构造 `CheckpointCallback`**，只保留 `LossCallback` 与 `MonitorCallback`；此时其余保存字段全部无效。
- **按步保存**：`on_step_end` 在 `state.global_step % save_interleaved_steps == 0` 时触发一次保存。
- **训练结束补存**：`on_train_end` 在训练结束时**额外保存一份最终权重**，确保末尾的训练成果不丢。
- **去重**：内部维护`_last_triggered_step`，若本次 step 与上次已保存的 step 重名则覆盖。
- **超额清理**：`save_max` 控制最多保留份数，超出时按时间删除最旧的目录，但仅会删除本轮次训练保存的权重。
- **路径校验**：`save_path` 为空会在构造回调时直接抛出 `ValueError("save_path must be provided for CheckpointCallback.")`，因此启用保存时必须配置 `save_path`。

> **输出目录结构**
>
> 每次保存会在 `save_path` 下生成一个按 step 命名的子目录，内含：
>
> - Safetensors 权重分片（模型分片，按 `no_save_optim` 决定是否含优化器分片）；
> - `common.json`：续训元信息（见[加载](#加载)一节）；
> - 分片布局元数据 `metadata.json`。

### 字段表

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_save` | bool | `True` | 是否启用权重保存；为 `False` 时不挂保存回调。 |
| `save_path` | str | `""` | 保存目录；启用保存时必填，为空报 `ValueError`。 |
| `save_max` | int | `5` | 保存权重最大数，超出时按时间删最旧，每次仅对当前训练保存的权重进行删除（特殊场景：若当前 step 已保存过 checkpoint，则覆盖该 step 对应目录） |
| `save_interleaved_steps` | int | `1000` | 保存间隔步数；`global_step` 为其整数倍时触发。 |
| `no_save_optim` | bool | `False` | 为 `True` 时仅保存模型权重，不保存优化器状态。 |
| `async_save` | bool | `False` | 为 `True` 时启用异步保存，落盘与计算重叠。 |
| `prefix` | str | `"checkpoint"` | 保存文件名前缀。 |
| `remove_redundancy` | bool | `False` | 为 `True` 时去除多卡分片间的冗余数据。 |
| `save_global_layout_cache` | bool | `True` | 为 `True` 时缓存多卡全局分片布局，避免每次保存重算分片元信息。 |

### 场景化配置

#### 场景一：基础定时保存

最常用的配置：每隔固定步数存一份，并保留最近若干份。适合大多数训练任务。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000   # 每 1000 步保存一次
  save_max: 5                    # 最多保留 5 份，超出删最旧
  prefix: "checkpoint"           # 保存的权重名前缀
  no_save_optim: False           # 同时保存优化器状态，便于整段续训（恢复权重、优化器状态和训练进度）
  async_save: False
  remove_redundancy: False
  save_global_layout_cache: True
```

> 适用：单数据源、单/多卡常规训练。代价：每次保存阻塞训练直到落盘完成；若 step 数很大、保存频繁，可考虑下面的异步保存。

#### 场景二：异步保存（降低保存阻塞）

当权重较大、同步落盘明显拖慢训练节奏时，开启 `async_save`。框架在保存前调用 `AsyncSaveManager.prepare_before_save`，将磁盘写入与后续训练计算重叠。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  async_save: True               # 启用异步保存
```

> 适用：大模型、保存频繁、磁盘较慢的场景。代价：保存在后台进行，会占用额外内存/线程资源；异常退出时最近一次异步保存可能尚未完成。

#### 场景三：仅存权重（不存优化器）

`no_save_optim=True` 时保存只包含模型权重，体积显著减小。适合只需要权重产物（如后续仅做微调）的场景。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  no_save_optim: True            # 不保存优化器状态
```

> **⚠️ 续训影响**
>
> `no_save_optim=True` 保存的权重缺少优化器状态，无法用于严格还原优化器动量和二阶矩的整段续训。若后续要从该权重进行续训，需在加载侧配合 `no_load_optim: True`。

#### 场景四：去冗余保存（减小占用）

多卡训练时，不同 rank 间存在重复的权重分片。`remove_redundancy=True` 在保存时去除这些冗余数据，减小磁盘占用。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  remove_redundancy: True        # 去除分片间冗余
```

> 适用：多卡分片、磁盘空间紧张的场景。注意去冗余依赖多卡分片元信息（`sharded_tensor_metas`，单卡恒为空），**单卡下该开关不生效（被静默忽略）**。

#### 场景五：多卡布局缓存

多卡场景下，保存需要先收集各 rank 的分片元信息（`sharded_tensor_metas`）。`save_global_layout_cache=True`（默认）会缓存这份元信息并在后续保存中复用；设为 `False` 则每次保存后清空缓存、下次重新计算。

```yaml
checkpoint:
  enable_save: True
  save_path: "./output/ckpt"
  save_interleaved_steps: 1000
  save_max: 5
  save_global_layout_cache: True # 缓存分片布局，加速后续保存
```

> 适用：分片布局在训练过程中固定不变的常规多卡训练，保持 `True` 可省去每次重算开销。仅当分片布局可能在训练中变化、需要每次重新采集时才设为 `False`。单卡场景不涉及分片，该字段无实际作用。

---

## 加载

### 加载流程

加载在 `Trainer.train` 启动阶段执行，权重加载目录取自 `load_path`。核心流程在 `Trainer._load_checkpoint`：

1. **读取元信息**：从 `load_path/common.json` 读出 `CommonInfo`（见下表）。
2. **恢复 step 与数据游标**：当 `no_load_optim=False`（整段续训）时，以 `common.json` 中的 `global_step` 作为续训起点；若当前 `global_batch_size` 与保存时不同，按比例缩放 `global_step = global_step * (旧 global_batch_size / 新 global_batch_size)`，再调用 `train_dataset.set_init_step(global_step)` 将数据集游标对齐到续训位置，并写回 `state.global_step`。
3. **加载权重（自动重分片）**：调用 `load_checkpoint`，内部用 `ReshardLoader` 处理分布式重分片，因此**保存与加载的并行布局可以不同**，框架会自动 reshard 到当前布局。
4. **均衡加载（可选）**：`load_balanced=True` 时走 `apply_balance_shard_strategy` 求出各 rank 间的冗余参数映射，再用 `single_parameter_broadcast` 在 rank 间广播参数，从而**消除冗余参数的重复读取**。
5. **优化器主权重刷新（可选）**：`no_load_optim=True` 时不加载优化器，加载完成后调用 `optimizer.reload_main_params_from_model()`，用刚加载的模型参数刷新 fp32 主权重，使主权重与模型参数对齐。

> **未配置加载目录**
>
> `load_path` 为空且未传入 `checkpoint_path` 时不加载任何权重，训练从随机初始化开始。

### `common.json` 记录的字段

#### common.json 示例

典型的 `common.json` 内容如下：

```json
{
  "epoch_num": 1,
  "step_num": 100,
  "global_step": 100,
  "loss_scale": "1.0",
  "global_batch_size": 2,
  "ckpt_status": null
}
```

`common.json` 由保存侧的 `CommonInfo` 写出（`mindformers/checkpoint/checkpoint.py`），续训时用于恢复 step 与数据游标：

| 字段 | 说明              | 续训中的作用                                               |
|---|-----------------|------------------------------------------------------|
| `epoch_num` | 当前训练所处 epoch    | 元信息记录                                                |
| `step_num` | 当前 epoch 内的步数   | 元信息记录                                                |
| `global_step` | 跨 epoch 的全局训练步数 | 续训起点；`global_batch_size` 变化时按比例缩放后驱动 `set_init_step` |
| `loss_scale` | 梯度放大系数          | 元信息记录                                                |
| `global_batch_size` | 多卡训练的全局批大小      | 与当前配置比较，决定是否缩放 `global_step`                         |
| `ckpt_status` | 权重健康状态标记        | 标识该权重是否健康，默认为 `null`，开启健康权重检测时记录权重健康状态。              |

### 字段表

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `load_path` | str | `""` | 加载目录；为空时不加载，从随机初始化开始。 |
| `no_load_optim` | bool | `False` | 为 `True` 时仅加载模型权重，不加载优化器状态，并刷新 fp32 主权重。 |
| `load_balanced` | bool | `False` | 为 `True` 时通过 shard 均衡 + 参数广播消除冗余参数的重复加载（多卡分片场景）。 |

`load_balanced` 适用于以下情况：

- TP（Tensor Parallel）规模较大；
- PP（Pipeline Parallel）规模较大；
- DP（Data Parallel）规模较大；
- checkpoint 文件数量较多；
- 加载阶段出现明显 I/O 瓶颈。

典型场景：

```text
TP=8
PP=8
DP=16
```

在大规模分布式训练中，checkpoint 文件通常分布在多个存储节点上。启用 `load_balanced` 后，可以减少部分 rank 的加载压力，提高整体恢复效率。

> **⚠️ load_worker_number 在动态图中未生效**
>
> `CheckpointConfig` 中虽声明了 `load_worker_number`（默认 `1`），但**动态图加载路径并未消费该字段**：`Trainer._load_checkpoint` 调用 `load_checkpoint` 时未传 `reshard_worker_num`，其始终使用默认值 `1`。
>
> 因此当前配置 `load_worker_number` 不会改变加载并行度，请勿依赖它加速读取。

### 场景化配置

#### 场景一：全量断点续训

恢复权重、优化器状态、`global_step` 与数据集游标，从中断处无缝继续。这是续训的默认形态。

```yaml
checkpoint:
  load_path: "./output/ckpt/checkpoint_5000"   # 指向某次保存的 step 目录
  no_load_optim: False                         # 加载优化器状态
  load_balanced: False
```

> 适用：训练中断后原地恢复。前提：被加载的权重保存时 `no_save_optim=False`（含优化器状态）。续训整体行为见 [断点续训](./resume_training.md)；数据游标恢复见 [数据集](./dataset.md)。

#### 场景二：微调仅加载权重

只加载模型权重，优化器从头初始化。适合从预训练权重出发做下游微调，不希望继承预训练阶段的优化器动量。

```yaml
checkpoint:
  load_path: "./pretrained/ckpt"
  no_load_optim: True              # 不加载优化器状态
```

> `no_load_optim=True` 时不会基于 `common.json` 缩放/续接 `global_step`，训练步从配置起点开始；加载完成后框架自动调用 `reload_main_params_from_model()` 刷新 fp32 主权重，避免主权重与模型参数错位。

#### 场景三：多卡均衡加载

`load_balanced=True` 时，框架用 shard 均衡策略求出各 rank 间的冗余参数映射，再通过参数广播让每个冗余参数只被读取一次，从而消除冗余参数的重复加载、降低分布式加载的显存与 IO。

```yaml
checkpoint:
  load_path: "./output/ckpt/checkpoint_5000"
  no_load_optim: False
  load_balanced: True              # 跨卡均衡加载
```

> **适用条件**
>
> `load_balanced` 走 `apply_balance_shard_strategy` 进行分片重分布，**仅在多卡分片场景下有意义**：单卡或无分片时不会带来收益。
>
> 并行维度与分片关系见 [分布式并行训练](./parallel_training.md)。

---

## 完整示例：训练中断后恢复训练

### 第一步：配置保存

```yaml
checkpoint:
  enable_save: True
  save_path: "./checkpoints"
  save_interleaved_steps: 1000
  no_save_optim: False
```

训练过程中生成：

```text
checkpoints/
├── iteration_00000100/
├── iteration_00000200/
└── latest_checkpointed_iteration.txt
```

---

### 第二步：训练中断

假设训练在 step=2300 时因节点故障退出。

最近一次成功保存的 checkpoint 为：

```text
iteration_00000200
```

---

### 第三步：配置加载

```yaml
checkpoint:
  load_path: "./checkpoints"
```

---

### 第四步：恢复过程

重新启动训练，系统自动完成：

1. 加载模型权重；
2. 加载优化器状态；
3. 恢复随机数状态；
4. 恢复 global_step；
5. 恢复数据集读取位置。

恢复完成后：

```text
global_step = 2000
```

训练将从 checkpoint 对应位置继续执行，而不是从头开始训练。

---

### 第五步：继续训练

```text
2000 -> 2001 -> 2002 -> ...
```

学习率调度、优化器动量以及数据集游标均保持连续。

## 相关文档

- 配置文件总览：[配置文件说明](./configuration.md)
- 断点续训流程与注意事项：[断点续训](./resume_training.md)
- 续训涉及的数据集游标恢复：[数据集](./dataset.md)
- 分片、并行维度与均衡加载背景：[分布式并行训练](./parallel_training.md)
- 训练全流程：[训练指南](../guide/training.md)
