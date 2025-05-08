# 启动任务

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/start_tasks.md)

## 概述

MindSpore Transformers提供了一键启动脚本`run_mindformer.py`和分布式任务拉起脚本`msrun_launcher.sh`。

- `run_mindformer.py`脚本用于在**单卡**上拉起任务，其提供了预训练、微调和推理任务的一键启动能力；
- `msrun_launcher.sh`脚本用于在**单机多卡**或**多机多卡**上拉起分布式任务，其通过[msrun](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)工具在每张卡上拉起任务。

## run_mindformer一键启动脚本

在MindSpore Transformers代码根目录下，使用Python执行`run_mindformer.py`脚本拉起任务，脚本支持的参数如下。**当可选参数未设置或设置为``None``时，取yaml配置文件中的同名配置**。

### 基础参数

|         参数          | 参数说明                                                                                                                       | 取值说明                                                    | 适用场景      |
|:-------------------:|:---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|-----------|
|     `--config`      | 任务yaml配置文件的路径。                                                                                                             | str，必选                                                  | 预训练/微调/推理 |
|      `--mode`       | 设置后端执行模式。                                                                                                                  | int，可选，`0`为GRAPH_MODE，`1`为PYNATIVE_MODE，当前仅支持GRAPH_MODE | 预训练/微调/推理 |
|    `--device_id`    | 设置执行设备ID，其值必须在可用设备范围内。                                                                                                     | int，可选                                                  | 预训练/微调/推理 |
|  `--device_target`  | 设置后端执行设备，MindSpore Transformers仅支持在`Ascend`设备上运行。                                                                          | str，可选                                                  | 预训练/微调/推理 |
|    `--run_mode`     | 设置模型的运行模式，可选`train`、`finetune`或`predict`。                                                                                  | str，可选                                                  | 预训练/微调/推理 |
| `--load_checkpoint` | 加载的权重文件或文件夹路径，详细使用方式参考[权重转换功能](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)。       | str，可选                                                  | 预训练/微调/推理 |
|  `--use_parallel`   | 是否开启并行模式。                                                                                                                  | bool，可选                                                 | 预训练/微调/推理 |
|   `--output_dir`    | 设置保存日志、权重、切分策略等文件的路径。                                                                                                      | str，可选                                                  | 预训练/微调/推理 |
|  `--register_path`  | 外挂代码所在目录的绝对路径。比如research目录下的模型目录。                                                                                          | str，可选                                                  | 预训练/微调/推理 |
|      `--seed`       | 设置全局种子，详情可参考[mindspore.set_seed](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。 | int，可选                                                  | 预训练/微调/推理 |

### 权重切分

|              参数              | 参数说明                                                                                                               | 取值说明                           | 适用场景      |
|:----------------------------:|:-------------------------------------------------------------------------------------------------------------------|--------------------------------|-----------|
| `--src_strategy_path_or_dir` | 权重的策略文件路径。                                                                                                         | str，可选                         | 预训练/微调/推理 |
|     `--auto_trans_ckpt`      | 是否开启在线权重自动转换功能，详情可参考[权重转换功能](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html)。 | bool，可选                        | 预训练/微调/推理 |
|  `--transform_process_num`   | 负责权重转换的进程数。                                                                                                        | int，可选                         | 预训练/微调/推理 |
|    `--only_save_strategy`    | 是否仅保存切分策略文件。                                                                                                       | bool，可选，为`true`时任务在保存策略文件后直接退出 | 预训练/微调/推理 |

### 训练

|               参数                | 参数说明                                                                                                                                              | 取值说明    | 适用场景   |
|:-------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------|---------|--------|
|      `--train_dataset_dir`      | 预训练/微调的数据集目录。                                                                                                                                     | str，可选  | 预训练/微调 |
|       `--resume_training`       | 是否开启断点续训功能，详情可参考[断点续训功能](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/resume_training.html#%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD)。 | bool，可选 | 预训练/微调 |
|           `--epochs`            | 训练轮次。                                                                                                                                             | int，可选  | 预训练/微调 |
| `--gradient_accumulation_steps` | 梯度累积步数。                                                                                                                                           | int，可选  | 预训练/微调 |
|         `--batch_size`          | 批处理数据的样本数。                                                                                                                                        | int，可选  | 预训练/微调 |
|         `--num_samples`         | 使用的数据集样本数量。                                                                                                                                       | int，可选  | 预训练/微调 |

### 推理

|           参数           | 参数说明                   | 取值说明                                                | 适用场景 |
|:----------------------:|:-----------------------|-----------------------------------------------------|------|
|    `--predict_data`    | 推理的输入数据。               | str，可选，可以是推理的输入（单batch推理）或包含多行文本的txt文件路径（多batch推理）。 | 推理   |
| `--predict_batch_size` | 多batch推理的batch_size大小。 | int，可选                                              | 推理   |
|     `--do_sample`      | 推理选择token时是否使用随机采样。    | int，可选，``True`` 表示使用随机采样，``False`` 代表使用贪心搜索。        | 推理   |

## 分布式任务拉起脚本

分布式任务拉起脚本`msrun_launcher.sh`位于`scripts/`目录下，可根据输入的参数自动使用[msrun](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)命令启动分布式多进程任务。该脚本有如下几种使用方式：

1. 默认使用单机8卡运行：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER]
```

2. 在单机上仅指定卡数快速运行：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM]
```

3. 单机自定义运行：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [MASTER_PORT] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
```

4. 多机自定义运行：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [LOCAL_WORKER] [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
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

## 任务启动教程

下面以Qwen2.5-0.5B微调为例，进行单卡、单机和多机任务使用方式说明。

### 单卡

在MindSpore Transformers代码根目录下执行Python脚本，进行单卡微调。命令中的路径需替换为真实路径。

```shell
python run_mindformer.py \
--register_path research/qwen2_5 \
--config finetune_qwen2_5_0_5b_8k.yaml \
--use_parallel False \
--run_mode finetune \
--train_dataset_dir ./path/alpaca-data.mindrecord
```

### 单机

在MindSpore Transformers代码根目录下执行msrun启动脚本，进行单机微调。命令中的路径需替换为真实路径。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
 --run_mode finetune \
 --train_dataset_dir ./path/alpaca-data.mindrecord "
```

### 多机

以Qwen2.5-0.5B为例，进行2机16卡微调。

1. 根据使用节点数等信息，修改相应的配置文件`research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml`：

```yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

> 如使用节点数和卡数改变需要修改`data_parallel`、 `model_parallel`、 `pipeline_stage`满足实际运行的卡数 `device_num=data_parallel×model_parallel×pipeline_stage`，同时满足`micro_batch_num >= pipeline_stage`。

2. 执行msrun启动脚本：

多机多卡执行脚本进行分布式任务需要分别在不同节点运行脚本，并将参数`MASTER_ADDR`设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，各个参数位置含义参见[msrun快速启动](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)。

```shell
# 节点0作为主节点, {ip_addr}处填写节点0实际ip, 总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --register_path research/qwen2_5 \
  --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
  --train_dataset_dir /{path}/wiki4096.mindrecord \
  --run_mode finetune" \
  16 8 {ip_addr} 8118 0 output/msrun_log False 300


# 节点1，{ip_addr}处填写节点0实际ip，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --register_path research/qwen2_5 \
  --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
  --train_dataset_dir /{path}/wiki4096.mindrecord \
  --run_mode finetune" \
  16 8 {ip_addr} 8118 1 output/msrun_log False 300
```
