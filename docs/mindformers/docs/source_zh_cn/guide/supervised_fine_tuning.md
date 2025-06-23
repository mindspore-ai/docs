# SFT微调

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/supervised_fine_tuning.md)

## 概述

SFT（Supervised Fine-Tuning，监督微调）采用有监督学习思想，是指在预训练模型的基础上，通过调整部分或全部参数，使模型更适应特定任务或数据集的过程。

MindSpore Transformers支持全参微调和LoRA高效微调两种SFT微调方式。全参微调是指在训练过程中对所有参数进行更新，适用于大规模数据精调，能获得最优的任务适应能力，但需要的计算资源较大。LoRA高效微调在训练过程中仅更新部分参数，相比全参微调显存占用更少、训练速度更快，但在某些任务中的效果不如全参微调。

## SFT微调的基本流程

结合实际操作，可以将SFT微调分解为以下步骤：

### 1. 权重准备

在微调之前，需要准备好预训练模型的权重文件。MindSpore Transformers提供加载 [safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)的能力，支持直接加载从 Hugging Face模型库中下载的模型权重。

### 2. 数据集准备

MindSpore Transformers微调阶段当前已支持[Hugging Face格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#huggingface%E6%95%B0%E6%8D%AE%E9%9B%86)以及[MindRecord格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#mindrecord%E6%95%B0%E6%8D%AE%E9%9B%86)的数据集。用户可根据任务需求完成数据准备。

### 3. 配置文件准备

微调任务通过[配置文件](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html)统一控制，用户可灵活调整[模型训练超参数](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/training_hyperparameters.html)。另外可以通过[分布式并行训练](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/parallel_training.html)、[内存优化特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/memory_optimization.html)以及[其它训练特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/other_training_features.html)对微调性能进行调优。

### 4. 启动训练任务

MindSpore Transformers提供[一键启动脚本](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/start_tasks.html)启动微调任务。训练过程中可结合[日志](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/logging.html)与[可视化工具](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html)监控训练情况。

### 5. 模型保存

训练过程中保存检查点或训练完成后，模型权重将保存至指定路径。当前支持保存为[Safetensors 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)或[Ckpt 格式](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html)，后续可以使用保存的权重进行续训或微调等。

### 6. 故障恢复

为应对训练中断等异常情况，MindSpore Transformers具备临终保存、自动恢复等[高可用特性](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/high_availability.html)，并支持[断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/resume_training.html)，提升训练稳定性。

## 使用MindSpore Transformers进行全参微调

### 选择预训练模型

MindSpore Transformers目前已经支持业界主流大模型，该实践流程选择Qwen2.5-7B模型为例。

### 下载模型权重

MindSpore Transformers提供加载Hugging Face模型权重的能力，支持直接加载从Hugging Face模型库中下载的模型权重。详细信息可以参考[MindSpore Transformers-Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)。

| 模型名称   |                Hugging Face权重下载链接           |
| :--------- | :--------------------------------------------: |
| Qwen2.5-7B | [Link](https://huggingface.co/Qwen/Qwen2.5-7B) |

### 数据集准备

MindSpore Transformers提供在线加载Hugging Face数据集的能力，详细信息可以参考[MindSpore Transformers-数据集-Hugging Face数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#huggingface%E6%95%B0%E6%8D%AE%E9%9B%86)。

本实践流程以[llm-wizard/alpaca-gpt4-data](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data)作为微调数据集为例。

|         数据集名称          | 适用阶段 |                              下载链接                               |
| :-------------------------: | :------: | :-----------------------------------------------------------------: |
| llm-wizard/alpaca-gpt4-data |   微调   | [Link](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data) |

### 执行微调任务

#### 单卡训练

首先准备配置文件，本实践流程以Qwen2.5-7B模型为例，提供了一个微调配置文件`finetune_qwen2_5_7b_8k_1p.yaml`，可以在[gitee仓库](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/supervised_fine_tuning/finetune_qwen2_5_7b_8k_1p.yaml)下载。

> 由于单卡显存有限，配置文件中的`num_layers`被设置为了4，仅作为示例使用。

然后根据实际情况修改配置文件中的参数，主要包括：

```yaml
load_checkpoint: '/path/to/Qwen2.5-7B/'                   # 预训练模型权重文件夹路径
...
train_dataset: &train_dataset
  ...
  data_loader:
    ...
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer:
          vocab_file: "/path/to/Qwen2.5-7B/vocab.json"    # 词表文件路径
          merges_file: "/path/to/Qwen2.5-7B/merges.txt"   # merges文件路径
```

执行`run_mindformer.py`启动单卡的微调任务，下面提供了一个使用示例：

启动命令如下：

```shell
python run_mindformer.py \
 --config /path/to/finetune_qwen2_5_7b_8k_1p.yaml \
 --register_path research/qwen2_5 \
 --use_parallel False \
 --run_mode finetune
```

参数说明：

```commandline
config：            模型的配置文件
use_parallel：      是否开启并行
run_mode：          运行模式，train：训练，finetune：微调，predict：推理
```

#### 单机训练

首先准备配置文件，本实践流程以Qwen2.5-7B模型为例，提供了一个微调配置文件`finetune_qwen2_5_7b_8k.yaml`，可以在[gitee仓库](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/supervised_fine_tuning/finetune_qwen2_5_7b_8k.yaml)下载。

然后根据实际情况修改配置文件中的参数，主要包括：

```yaml
load_checkpoint: '/path/to/Qwen2.5-7B/'                   # 预训练模型权重文件夹路径
...
train_dataset: &train_dataset
  ...
  data_loader:
    ...
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer:
          vocab_file: "/path/to/Qwen2.5-7B/vocab.json"    # 词表文件路径
          merges_file: "/path/to/Qwen2.5-7B/merges.txt"   # merges文件路径
```

执行以下msrun启动脚本，进行8卡分布式训练：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config /path/to/finetune_qwen2_5_7b_8k.yaml \
 --use_parallel True \
 --run_mode finetune" 8
```

参数说明：

```commandline
config：            模型的配置文件
use_parallel：      是否开启并行
run_mode：          运行模式，train：训练，finetune：微调，predict：推理
```

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

#### 多机训练

多机多卡微调任务与启动预训练类似，可参考[多机多卡的预训练命令](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html#%E5%A4%9A%E6%9C%BA%E8%AE%AD%E7%BB%83)。

首先对配置文件进行修改，这里需要针对不同的机器数量进行设置：

```yaml
parallel_config:
  data_parallel: ...
  model_parallel: ...
  pipeline_stage: ...
  context_parallel: ...
```

并对命令进行如下修改：

1. 增加启动脚本入参`--config /path/to/finetune_qwen2_5_7b_8k.yaml`加载预训练权重。
2. 设置启动脚本中的`--run_mode finetune`，run_mode表示运行模式，train：训练，finetune：微调，predict：推理。

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

## 使用MindSpore Transformers进行LoRA高效微调

MindSpore Transformers支持配置化使能LoRA微调，无需对每个模型进行代码适配，而仅需修改全参微调的YAML配置文件中的模型配置，添加 `pet_config` 高效微调配置，即可使用其进行LoRA高效微调任务。以下展示了Llama2模型LoRA微调的YAML配置文件中的模型配置部分，并对 `pet_config` 参数进行了详细说明。

### LoRA 原理简介

LoRA通过将原始模型的权重矩阵分解为两个低秩矩阵来实现参数量的显著减少。例如，假设一个权重矩阵W的大小为$m \times n$，通过LoRA，该矩阵被分解为两个低秩矩阵A和B，其中A的大小为$m \times r$，B的大小为$r \times n$（$r$远小于$m$和$n$）。在微调过程中，仅对这两个低秩矩阵进行更新，而不改变原始模型的其他部分。

这种方法不仅大幅度降低了微调的计算开销，还保留了模型的原始性能，特别适用于数据量有限、计算资源受限的环境中进行模型优化，详细原理可以查看论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 。

### 修改配置文件

基于全参微调的配置文件，我们需要在模型配置中添加LoRA相关的参数，并将其重命名为`fine_tune_qwen2_5_7b_8k_lora.yaml`。以下是一个示例配置片段，展示了如何在Qwen2.5-7B模型的配置文件中添加LoRA微调的相关参数：

```yaml
# model config
model:
  model_config:
    ...
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
```

### pet_config 参数详解

在 model_config 中，pet_config 是LoRA微调的核心配置部分，用于指定LoRA的相关参数。具体参数说明如下：

- **pet_type:** 指定参数高效微调技术（PET，Parameter-Efficient Tuning）的类型为LoRA。这意味着在模型的关键层中会插入LoRA模块，以减少微调时所需的参数量。
- **lora_rank:** 定义了低秩矩阵的秩值。秩值越小，微调时需要更新的参数越少，从而减少计算资源的占用。这里设为16是一个常见的平衡点，在保持模型性能的同时，显著减少了参数量。
- **lora_alpha:** 控制LoRA模块中权重更新的缩放比例。这个值决定了微调过程中，权重更新的幅度和影响程度。设为16表示缩放幅度适中，有助于稳定训练过程。
- **lora_dropout:** 设置LoRA模块中的dropout概率。Dropout是一种正则化技术，用于减少过拟合风险。设置为0.05表示在训练过程中有5%的概率会随机“关闭”某些神经元连接，这在数据量有限的情况下尤为重要。
- **target_modules:** 通过正则表达式指定LoRA将应用于模型中的哪些权重矩阵。在Llama中，这里的配置将LoRA应用于模型的自注意力机制中的Query（wq）、Key（wk）、Value（wv）和Output（wo）矩阵。这些矩阵在Transformer结构中扮演关键角色，插入LoRA后可以在减少参数量的同时保持模型性能。

### Qwen2.5-7B 的 LoRA 微调示例

LoRA微调过程中使用的数据集可以参考全参微调部分的[数据集准备](#数据集准备)章节。

以 Qwen2.5-7B 为例，可以执行以下 msrun 启动脚本，进行 8 卡分布式微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config /path/to/finetune_qwen2_5_7b_8k_lora.yaml \
 --use_parallel True \
 --run_mode finetune" 8
```
