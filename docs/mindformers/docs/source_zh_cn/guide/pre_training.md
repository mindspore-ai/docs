# 预训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/pre_training.md)

## 概述

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。通过预训练，模型可以学习到词汇、句法和语义等层面的知识，这些知识在下游任务中通过微调得到应用，从而优化特定任务的性能。MindSpore Transformers框架的预训练目标是帮助开发者快速、便捷地构建和训练基于Transformer架构的预训练模型。

## MindSpore Transformers 的预训练流程

结合实际操作，预训练的基本流程可以分解为以下步骤：

### 1. 数据集准备

MindSpore Transformers 预训练阶段当前已支持[Megatron 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)和[MindRecord格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#mindrecord%E6%95%B0%E6%8D%AE%E9%9B%86)的数据集。用户可根据任务需求完成数据准备。

### 2. 配置文件准备

预训练任务通过[配置文件](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html)统一控制，用户可灵活调整[模型训练超参数](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/training_hyperparameters.html)。另外可以通过[分布式并行训练](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/parallel_training.html)、[内存优化特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/memory_optimization.html)以及[其它训练特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/other_training_features.html)对预训练性能进行调优。

### 3. 启动训练任务

MindSpore Transformers 提供[一键启动脚本](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/start_tasks.html)启动预训练任务。训练过程中可结合[日志](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/logging.html)与[可视化工具](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html)监控训练情况。

### 4. 模型保存

在中间保存检查点或训练完成后，模型权重将保存至指定路径。当前支持保存为[Ckpt 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html)或[Safetensors 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)，后续可以使用保存的权重进行续训或微调等。

### 5. 故障恢复

为应对训练中断等异常情况，MindSpore Transformers 具备临终保存、自动恢复等[高可用特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/high_availability.html)，并支持[断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/resume_training.html)，提升训练稳定性。

## 基于 MindSpore Transformers 的预训练实践

MindSpore Transformers 目前已经支持业界主流大模型，本实践流程选择以 DeepSeek-V3-671B 展示单机训练和多机训练。

### 数据集准备

MindSpore Transformers 目前已经支持加载 Megatron 数据集，该数据集通常经过预处理，序列化为二进制格式（例如`.bin`或`.idx`文件），并配套特定索引机制，便于在分布式集群环境下高效并行加载与数据切分。

- 数据集下载：[wikitext-103数据集](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- 分词模型下载：分词模型[tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

### 数据预处理

数据集处理可参考[Megatron数据集-数据预处理](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)

- 生成Megatron BIN格式文件

   将数据集文件`wiki.train.tokens`和分词模型文件`tokenizer.json`放置在`../dataset`下。

   使用以下命令将数据集文件转换为BIN格式文件。

   ```shell
   cd $MINDFORMERS_HOME
   python research/deepseek3/wikitext_to_bin.py \
    --input ../dataset/wiki.train.tokens \
    --output-prefix ../dataset/wiki_4096 \
    --vocab-file ../dataset/tokenizer.json \
    --seq-length 4096 \
    --workers 1
   ```

- 构建Megatron BIN数据集模块

   执行如下命令构建Megatron BIN数据集模块。如使用提供的镜像请跳过此操作。

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

   其中，`$MINDFORMERS_HOME` 指 Mindspore Transformers 源代码所在的目录。

## 执行预训练任务

### 单机训练

通过指定模型路径和配置文件[pretrain_deepseek3_671b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml)以msrun的方式启动[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/dev/run_mindformer.py)脚本，进行8卡分布式训练。

默认配置中的模型层数、隐藏维度等参数较大，适用于多机大规模分布式训练，无法直接在单机环境启动预训练，需要参考[DeepSeek-V3-修改配置](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/README.md#%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE)修改配置文件。

启动详细介绍详见[拉起任务](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/README.md#%E6%8B%89%E8%B5%B7%E4%BB%BB%E5%8A%A1)，启动命令如下：

```shell
cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/deepseek3 \
 --config research/deepseek3/deepseek3_671b/pretrain_deepseek3_1b.yaml"
```

   其中，
   - `register_path`：  模型地址，即模型实现文件所在目录
   - `config`：         模型的配置文件，文件在 MindSpore Transformers 代码仓中 config 目录下

任务执行完成后，在 mindformers/output 目录下，会生成 checkpoint 文件夹，同时模型文件(`.safetensors`)会保存在该文件夹下。

### 多机训练

如果服务器资源充足，可以参考如下方式拉起多台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的Rank序号，从`0`到`1023`。

```shell
master_ip=192.168.1.1
node_rank=0

cd $MINDFORMERS_HOME
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/deepseek3 \
 --config research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml" \
 1024 8 $master_ip 8118 $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号。

**注意**： 在多机分布式训练的过程中，可能会遇到一些性能问题。为了确保训练过程的高效性和稳定性，建议参考[大模型性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html)，进行必要的性能优化和调整。

## 更多信息

更多关于不同模型的训练示例，请访问[MindSpore Transformers已支持模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/introduction/models.html)。