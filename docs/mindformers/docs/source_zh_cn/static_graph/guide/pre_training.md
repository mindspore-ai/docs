<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} 已废弃（Deprecated）
:class: warning

本页属于 **静态图（GRAPH_MODE）实现** 章节，已标记为废弃。新特性优先在「r2.0.0 动态图实现」章节演进，请优先查阅动态图相关文档。
```

# 预训练实践

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/static_graph/guide/pre_training.md)

## 概述

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。通过预训练，模型可以学习到词汇、句法和语义等层面的知识，这些知识在下游任务中通过微调得到应用，从而优化特定任务的性能。MindSpore Transformers框架的预训练目标是帮助开发者快速、便捷地构建和训练基于Transformer架构的预训练模型。

## MindSpore Transformers 的预训练流程

结合实际操作，预训练的基本流程可以分解为以下步骤：

### 1. 数据集准备

MindSpore Transformers 预训练阶段当前已支持[Megatron 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)和[MindRecord格式](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/dataset.html#mindrecord%E6%95%B0%E6%8D%AE%E9%9B%86)的数据集。用户可根据任务需求完成数据准备。

### 2. 配置文件准备

预训练任务通过[配置文件](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/configuration.html)统一控制，用户可灵活调整[模型训练超参数](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/training_hyperparameters.html)。另外可以通过[分布式并行训练](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/parallel_training.html)、[内存优化特性](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/memory_optimization.html)以及[其它训练特性](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/other_training_features.html)对预训练性能进行调优。

### 3. 启动训练任务

MindSpore Transformers 提供[一键启动脚本](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/start_tasks.html)启动预训练任务。训练过程中可结合[日志](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/logging.html)与[可视化工具](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/monitor.html)监控训练情况。

### 4. 模型保存

在中间保存检查点或训练完成后，模型权重将保存至指定路径。当前支持保存为[Ckpt 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/ckpt.html)或[Safetensors 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/safetensors.html)，后续可以使用保存的权重进行续训或微调等。

### 5. 故障恢复

为应对训练中断等异常情况，MindSpore Transformers 具备临终保存、自动恢复等[训练高可用](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/high_availability.html)特性，并支持[断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/resume_training.html)，提升训练稳定性。

## 基于 MindSpore Transformers 的预训练实践

MindSpore Transformers 目前已经支持业界主流大模型，本实践流程选择以 Qwen3-32B 展示单机训练和多机训练。更多训练示例与各场景下的推荐配置，请参阅 [模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/models.html)。

### 数据集准备

MindSpore Transformers 目前已经支持加载 Megatron 数据集，该数据集通常经过预处理，序列化为二进制格式（例如`.bin`或`.idx`文件），并配套特定索引机制，便于在分布式集群环境下高效并行加载与数据切分。

- 数据集下载：[wikitext-103数据集](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- 分词模型下载：分词模型[tokenizer.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer.json)

### 数据预处理

MindSpore Transformers 预训练阶段当前已支持[Megatron格式的数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)。用户可以参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/dataset.html)章节，使用 MindSpore 提供的工具将原始数据集转换为 Megatron 格式。

制作Megatron格式数据集，需要经过两个步骤。首先将原始文本数据集转换为jsonl格式数据，然后使用MindSpore Transformers提供的脚本将jsonl格式数据转换为Megatron格式的.bin和.idx文件。

- `wiki.train.tokens` 转为 `jsonl`格式数据

  用户需要**自行将`wiki.train.tokens`数据集处理成jsonl格式的文件**。作为参考，[社区issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY)提供了一个转换方案，用户需要根据实际需求自行开发和验证转换逻辑。

  下面是jsonl格式文件的示例：

  ```json
  {"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
  {"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
  ...
  ```

- `jsonl`格式数据 转为 `bin`格式数据

  MindSpore Transformers提供了数据预处理脚本`toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py`用于将jsonl格式的原始文本语料转换成.bin或.idx文件。

  > 这里需要提前下载[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer.json)模型的tokenizer文件。

  示例：

  ```shell
  python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/wiki103-megatron \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-dir /path/to/Qwen3-32B # 其他规格的模型可以调整为对应的tokenizer路径
  ```

  运行完成后会生成`/path/to/wiki103-megatron_text_document.bin`和`/path/to/wiki103-megatron_text_document.idx`文件。

  填写数据集路径时需要使用`/path/to/wiki103-megatron_text_document`，不需要带后缀名。

## 执行预训练任务

### 单机训练

通过指定模型路径和配置文件[pretrain_qwen3_32b_4k.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/pretrain_qwen3_32b_4k.yaml)，修改配置后以msrun的方式启动[run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，进行8卡分布式训练。

仓上提供的配置为32B模型，参数量较大，无法直接在单机环境启动预训练。本例中缩减模型规模至0.6B，以演示单机训练。修改配置文件中的如下参数，其余参数保持不变：

```yaml
# model_config
model:
  model_config:
    hidden_size: 1024
    num_attention_heads: 16
    num_hidden_layers: 28
```

启动命令如下：

```shell
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 2 \
--parallel_config.pipeline_stage 4 \
--parallel_config.micro_batch_num 4"
```

其中：

- `config`：模型的配置文件，文件在MindSpore Transformers代码仓中config目录下。
- `parallel_config.data_parallel`：设置数据并行数。
- `parallel_config.model_parallel`：设置模型并行数。
- `parallel_config.pipeline_stage`：设置流水线并行数。
- `parallel_config.micro_batch_num`：设置流水线并行的微批次大小，在`parallel_config.pipeline_stage`大于1时，应满足`parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage`。

启动详细介绍详见[启动预训练任务](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/README.md#3-启动预训练任务)。

任务执行完成后，在 mindformers/output 目录下，会生成 checkpoint 文件夹，同时模型文件（`.safetensors`）会保存在该文件夹下。

### 多机训练

如果服务器资源充足，可以参考如下方式拉起多台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`1023`。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时，请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。

**注意**：在多机分布式训练的过程中，可能会遇到一些性能问题。为了确保训练过程的高效性和稳定性，建议参考[大模型性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/performance_optimization.html)，进行必要的性能优化和调整。
