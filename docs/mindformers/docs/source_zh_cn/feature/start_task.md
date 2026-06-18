# 启动任务

MindSpore Transformers 动态图（PyNative）训练提供了一键启动脚本 `run_mindformer.py` 和分布式任务拉起脚本 `msrun_launcher.sh`。

- `run_mindformer.py`脚本用于在**单卡**上拉起任务，其提供了预训练任务的一键启动能力；
- `msrun_launcher.sh`脚本用于在**单机多卡**或**多机多卡**上拉起分布式任务，其通过[msrun](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)工具在每张卡上拉起任务。

## 一、run_mindformer 一键启动脚本

在 MindSpore Transformers 代码根目录下，使用 Python 执行 `run_mindformer.py` 脚本拉起任务，脚本支持的参数如下。**当可选参数未设置或设置为 ``None`` 时，取 YAML 配置文件中的同名配置**。

### 基础参数

|          参数           | 参数说明                                                                                                                        | 取值说明                                | 适用场景 |
|:---------------------:|:----------------------------------------------------------------------------------------------------------------------------|-------------------------------------|------|
|      `--config`       | 任务yaml配置文件的路径。                                                                                                              | str，必选                              | 预训练  |
|       `--mode`        | 设置后端执行模式。                                                                                                                   | int，必选，需要设置 `1` 指定使用 PYNATIVE_MODE。 | 预训练  |
|     `--run_mode`      | 设置模型的运行模式，可选`train`。                                                                                                        | str，可选                              | 预训练  |
|   `--use_parallel`    | 是否开启并行模式。                                                                                                                   | bool，可选                             | 预训练  |
|    `--output_dir`     | 设置保存日志、权重、切分策略等文件的路径。                                                                                                       | str，可选                              | 预训练  |
|       `--seed`        | 设置全局种子，详情可参考[mindspore.set_seed](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。  | int，可选                              | 预训练  |

## 二、分布式任务拉起脚本

分布式任务拉起脚本 `msrun_launcher.sh` 位于 `scripts/` 目录下，可根据输入的参数自动使用[msrun](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)命令启动分布式多进程任务。该脚本有如下几种使用方式：

1. 默认使用单机8卡运行：

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER]
    ```

2. 在单机上仅指定卡数快速运行：

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM]
    ```

3. 单机自定义运行：

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [MASTER_PORT] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
    ```

4. 多机自定义运行：

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [LOCAL_WORKER] [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
    ```

脚本的参数说明如下：

|         参数         | 参数说明                          | 取值说明                              |
|:------------------:|:------------------------------|-----------------------------------|
|  `EXECUTE_ORDER`   | 要分布式执行的Python脚本命令参数。          | str，必选，设置为包含要执行的Python脚本和脚本参数的字符串 |
|    `WORKER_NUM`    | 参与分布式任务的Worker进程总数。           | int，可选，默认值：`8`                    |
|   `LOCAL_WORKER`   | 当前节点上拉起的Worker进程数。            | int，可选，默认值：`8`                    |
|   `MASTER_ADDR`    | 指定Scheduler的IP地址或者主机名。        | str，可选，默认值：`"127.0.0.1"`          |
|   `MASTER_PORT`    | 指定Scheduler绑定端口号。             | int，可选，默认值：`8118`                 |
|    `NODE_RANK`     | 当前节点的索引。                      | int，可选，默认值：`0`                    |
|     `LOG_DIR`      | Worker以及Scheduler日志输出路径。      | str，可选，默认值：`"output/msrun_log"`   |
|       `JOIN`       | msrun是否等待Worker以及Scheduler退出。 | bool，可选，默认值：`False`               |
| `CLUSTER_TIME_OUT` | 集群组网超时时间，单位为秒。                | int，可选，默认值：`7200`                 |

## 三、任务启动教程

下面以动态图场景为例，进行单卡、单机和多机任务使用方式说明。

### 单卡启动

可通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 选择具体某张卡执行任务：

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 # 指定选择 7 号卡执行任务
```

> 如不通过环境变量指定卡，则默认选择 0 卡执行单卡任务。

在 MindSpore Transformers 代码根目录下执行 Python 脚本，进行单卡训练。单卡启动命令示例如下：

```bash
python run_mindformer.py --config /path/to/your.yaml --mode 1
```

> `ASCEND_RT_VISIBLE_DEVICES=0` 表示只让进程看到 0 号物理卡；若想使用 3 号卡则写 `=3`。多卡时可写成 `=0,1,2,3` 这样的列表形式。

### 单机多卡启动

在 MindSpore Transformers 代码根目录下执行 msrun 启动脚本，进行单机训练任务。

并行切分维度（FSDP/HSDP、TP、CP、PP、EP 等）由配置文件中的 `parallelism` 段决定。下面以单机 8 卡场景为例，切分设置为 fsdp8，其对应的 YAML 文件字段如下示例：

```yaml
training:                    # 训练超参数（命名风格沿用 torchtitan）
  steps: 10                  # 总训练步数
  local_batch_size: 1        # 每张卡（per-rank）的批大小
  global_batch_size: 8       # 全局每步样本数 = 各 DP 分片之和（用于计算梯度累积步数）
  seed: 42

parallelism:                 # 并行切分
  data_parallel_shard: -1    # -1 表示自动根据剩余卡数推导 FSDP 分片度
  data_parallel_shard_strategy: "optim_grads_params"
  tensor_parallel: 1         # TP
  context_parallel: 1        # CP
  pipeline_parallel: 1       # PP
  expert_parallel: 1         # EP（专家并行，不计入 world_size 乘积，见下方说明）

train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    config:                    # 完整字段说明见「数据集」文档
      seq_length: 4096
      split: "1, 0, 0"
      eod: 1
      pad: -1
      create_attention_mask: False
      create_compressed_eod_mask: False
      data_path:
        - '1'
        - "/path/megatron_data_text_document"
  drop_remainder: True
  num_parallel_workers: 8
```

配置好 YAML 项后，使用 `msrun_launcher.sh` 脚本拉起任务：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config /path/to/your.yaml \
--mode 1" 8
```

**配置切分如何对应 8 卡？** 框架首先计算数据并行度，再将其拆分为 `dp_replicate * dp_shard`：

- `data_parallel = world_size / (tensor_parallel * pipeline_parallel * context_parallel)` = `8 / (1*1*1)` = `8`
- `data_parallel_shard = -1` 时，自动设置 `dp_replicate = 1`、`dp_shard = data_parallel = 8`，即 8 张卡全部用于 FSDP 分片
- 梯度累积步数 `= global_batch_size / (data_parallel_shard * local_batch_size)` = `8 / (8*1)` = `1`

> **world_size 校验：哪些并行度相乘等于卡数**
>
> 框架要求 **`dp_replicate * dp_shard * cp * tp * pp == world_size`**（即 `--worker_num`），若不满足会直接报错。
>
> **`expert_parallel`（EP）不计入该乘积**——EP 是在数据并行/张量并行维度内部对专家进行的再切分，不应将其乘入 `world_size`。例如在 2 卡上设置 `expert_parallel: 2` 时，仍按 `dp_shard=2、tp=1、pp=1、cp=1`（乘积=2=world_size）通过校验，EP 仅决定专家如何在这 2 张卡之间分布。

### 多机多卡启动

在多机场景下，**每个节点需要各自执行一条 `msrun_launcher.sh` 命令**。通过 `NODE_RANK` 区分不同节点（主节点为 0、从节点为 1、2），其余参数需满足以下要求：

- `WORKER_NUM`、`MASTER_ADDR`、`MASTER_PORT` **所有节点必须完全一致**
- `LOCAL_WORKER` 通常也保持一致，**仅当各节点卡数不同时才需要逐节点设置不同值**
- `NODE_RANK` **逐节点递增**，用于稳定地为各节点分配 rank id

以 2 机 16 卡（每机 8 卡）为例，假设主节点 IP 为 `192.168.1.1`：

**主节点（节点 0）：**

```bash
# 节点0作为主节点, 总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config /path/to/your.yaml \
  --mode 1" \
  16 8 192.168.1.1 8118 0 output/msrun_log False 7200
```

**从节点（节点 1）：**

```bash
# 节点1，主节点ip为192.168.1.1，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config /path/to/your.yaml \
  --mode 1" \
  16 8 192.168.1.1 8118 1 output/msrun_log False 7200
```

两条命令**仅有 `NODE_RANK` 不同**（分别为 0 和 1），其余参数完全一致。若各节点卡数不同，则需要额外调整对应节点的 `LOCAL_WORKER`，但 `WORKER_NUM` 仍需保持为全局总卡数。

> **不设 NODE_RANK 的风险**
>
> 若不设置 `NODE_RANK`，MindSpore 会自动分配 rank id（同节点内 rank 连续），但节点间的编号依赖于各节点的拉起顺序，存在错配与不可复现的风险。因此**在多机场景下务必显式设置 `NODE_RANK`**。

启动多机任务前请确保：各节点间网络互通、`MASTER_PORT` 未被占用、各节点使用**相同的代码、配置与数据集路径**；对于大集群，若组网较慢可调大 `CLUSTER_TIME_OUT`。

## 四、自定义脚本启动

PyNative 训练支持在自定义脚本里直接构建并启动训练器。例如可编写如下自定义脚本，指定 `config` 为对应 YAML 地址后，即可启动 PyNative 训练：

```python
from mindformers.pynative.trainer import Trainer as PynativeTrainer

def main():
    my_config = "/path/to/your.yaml"

    trainer = PynativeTrainer(config=my_config)
    trainer.train()

if __name__ == "__main__":
    main()
```

> **自定义脚本和 run_mindformer 统一入口的区别：**
>
> - `run_mindformer.py --mode 1` 与「自定义脚本」**最终都构建同一个 `PynativeTrainer` 并调用 `train()`**，运行行为一致。
> - 区别仅在于「谁来解析命令行、装配配置」：统一入口由 `run_mindformer.py` 完成；自定义脚本由您自己指定 `config` 及其他入参完成，更灵活，便于在脚本里修改配置、进行测试。
> - 仓库中的测试用例（如 `tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3/run_deepseek3.py`）即为自定义脚本，因此按照仓库命令操作时看到的不是 `--mode 1`，而是 `bash scripts/msrun_launcher.sh "run_deepseek3.py --config xxx.yaml"`。

## 五、启动期排障

初次启动失败大多集中在：组网、端口、单卡与多卡差异等方面，可参考下表快速定位问题：

| 现象                        | 排查方向                                                                                                            |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| 进程卡在初始化阶段、报「等不到足够 worker」 | 节点未全部拉起或网络不通；检查各节点 `WORKER_NUM` 是否一致、`MASTER_ADDR/MASTER_PORT` 是否正确；大集群可适当调大 `CLUSTER_TIME_OUT`（负数为无限等待）。       |
| 报端口被占用 / scheduler 启动失败   | `MASTER_PORT` 被占用或不在 `1024–65535` 范围内；更换一个未占用端口，并确保所有节点使用同一端口。                                                  |
| 多机 rank 错配、训练 hang 在通信阶段  | 未设置 `NODE_RANK`；请按主节点 0、从节点 1 的顺序显式设置。                                                                          |
| HCCL/组网超时、跨机集合通信报错        | 检查节点互通性与 HCCL 配置；适当调大 `CLUSTER_TIME_OUT`；确认各节点配置与数据路径一致。                                                        |
| 单卡能跑通、多卡运行失败              | 多为并行配置或集合通信问题：首先核对 `dp_replicate * dp_shard * cp * tp * pp == world_size`（EP 不计入），再检查是否存在多机 `NODE_RANK` 或组网问题。  |

**查看日志：** `msrun` 会为每个 worker 在 `LOG_DIR` 目录下生成独立日志文件（scheduler 与各 worker 分别记录），初次启动报错时应直接查看对应 worker 的日志而非前台聚合输出；当设置 `JOIN=True` 时，`msrun` 会解析日志并回收退出码。

## 相关文档

- 配置文件字段与 schema：[配置文件说明](./configuration.md)
- 并行维度、world_size 推导与批大小：[分布式并行训练](./parallel_training.md)
- 端到端启动示例：[快速开始](../quick_start/quick_start.md)
