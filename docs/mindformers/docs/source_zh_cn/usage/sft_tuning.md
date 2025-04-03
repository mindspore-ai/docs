# SFT微调

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/usage/sft_tuning.md)

## 概述

SFT（Supervised Fine-Tuning，监督微调）采用有监督学习思想，是指在源数据集上进行预训练，得到一个原始模型，然后在一个新的数据集上对原始模型进行参数调整，得到新的模型，使其在新的任务上有更好的表现。

## SFT微调的基本流程

SFT微调整体包含以下几个部分：

- **预训练：**
  首先需要在一个较大规模的数据集上训练一个神经网络模型，比如针对大语言模型，通常是在大量未标记的文本数据上进行，预训练阶段的目标是使模型获取通用的知识和理解能力。
- **微调：**
  结合目标任务，用新的训练数据集对已经得到的预训练模型进行微调。在微调过程中，通过反向传播可以对原始模型的全部参数或者部分参数进行优化，使模型在目标任务上取得更好的效果。
- **评估：**
  经过微调之后会得到一个新的模型，可以用目标任务的评测数据集对微调模型进行评估，从而得到微调模型在目标任务上的性能指标。

结合实际操作，可以将SFT微调分解为以下步骤：

1. **选择预训练模型：**
   选择一个预训练的语言模型，如GPT-2、Llama2等。预训练模型通常是在大型文本语料库上进行过训练，以学习语言的通用表示。
2. **下载模型权重：**
   针对选择的预训练模型，可以从HuggingFace模型库中下载预训练的权重。
3. **模型权重转换：**
   结合自己所要使用的框架，对已经下载的HuggingFace权重进行权重转换，比如转换为MindSpore框架所支持的ckpt权重。
4. **数据集准备：**
   结合微调的目标，选择用于微调任务的数据集，针对大语言模型，微调数据集一般是包含文本和标签的数据，比如alpaca数据集。同时在使用数据集时，需要对数据做相应的预处理，比如使用MindSpore框架时，需要将数据集转换为MindRecord格式。
5. **执行微调任务：**
   使用微调任务的数据集对预训练模型进行训练，更新模型参数，如果是全参微调则会对所有参数进行更新，微调任务完成后，便可以得到新的模型。

## 基于MindSpore Transformers的全参微调实践

### 选择预训练模型

MindSpore Transformers目前已经支持业界主流大模型，该实践流程选择Llama2-7B模型SFT微调为例。

### 下载模型权重

MindSpore Transformers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过模型权重转换后进行使用。

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

| 模型名称      |                                                 MindSpore权重                                                  |                                        HuggingFace权重                                        |
|:----------|:------------------------------------------------------------------------------------------------------------:| :---------------------------------------------------------------------------------------------: |
| Llama2-7B |  [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)      | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

> Llama2的所有权重都需要通过向Meta[提交申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)来获取，如有需要请自行申请。

### 模型权重转换

以[Llama2-7B模型](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)为例，原始的HuggingFace权重文件主要包含：<br>

- `config.json`：模型架构的主要配置信息<br>
- `generation_config.json`：文本生成相关的配置信息<br>
- `safetensors文件`：模型权重文件<br>
- `model.safetensors.index.json`：safetensors模型参数文件索引和描述模型切片的json文件<br>
- `bin文件`：pytorch的模型权重文件<br>
- `pytorch_model.bin.index.json`：pytorch索引和描述模型切片的json文件<br>
- `tokenizer.json`：分词器的词汇配置文件<br>
- `tokenizer.model`：模型的分词器<br>

MindSpore Transformers提供权重转换脚本，通过执行[convert_weight.py转换脚本](https://gitee.com/mindspore/mindformers/blob/v1.3.2/convert_weight.py)，可以将HuggingFace的权重转换为完整的ckpt权重。

```bash
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME
```

参数说明:

```commandline
model:       模型名称（其他模型请参考模型说明文档）
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

### 数据集准备

MindSpore Transformers提供**WikiText2**作为预训练数据集，**alpaca**作为微调数据集。

| 数据集名称     |                 适用模型                  |   适用阶段    |                                                                            下载链接                                                                            |
|:----------|:-------------------------------------:|:---------:| :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| alpaca    | Llama2-7B<br>Llama2-13B<br>Llama2-70B |    微调     |                   [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                   |

以alpaca数据集为例，下载数据集后需要对数据集进行预处理。预处理中所用的`tokenizer.model`可以参考模型权重下载进行下载。

**alpaca 数据预处理**

1. 执行MindSpore Transformers中的[alpaca_converter.py脚本](https://gitee.com/mindspore/mindformers/blob/v1.3.2/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py)，将数据集转换为多轮对话格式。

    ```bash
    python alpaca_converter.py \
      --data_path /{path}/alpaca_data.json \
      --output_path /{path}/alpaca-data-conversation.json
    ```

    参数说明：

    ```commandline
    data_path:   输入下载的文件路径
    output_path: 输出文件的保存路径
    ```

2. 执行MindSpore Transformers中的[llama_preprocess.py脚本](https://gitee.com/mindspore/mindformers/blob/v1.3.2/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py)，将数据转换为MindRecord格式。该操作依赖fastchat工具包解析prompt模板, 请提前安装fastchat >= 0.2.13。

    ```bash
    python llama_preprocess.py \
      --dataset_type qa \
      --input_glob /{path}/alpaca-data-conversation.json \
      --model_file /{path}/tokenizer.model \
      --seq_length 4096 \
      --output_file /{path}/alpaca-fastchat4096.mindrecord
    ```

    参数说明：

    ```commandline
    dataset_type: 预处理数据类型
    input_glob:   转换后的alpaca的文件路径
    model_file:   模型tokenizer.model文件路径
    seq_length:   输出数据的序列长度
    output_file:  输出文件的保存路径
    ```

### 执行微调任务

#### 单机训练

以Llama2-7B为例，执行msrun启动脚本，进行8卡分布式训练，启动命令如下：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b.yaml \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --use_parallel True \
 --run_mode finetune" 8
```

参数说明：

```commandline
config：            模型的配置文件，文件在MindSpore Transformers代码仓中config目录下
load_checkpoint：   checkpoint文件的路径
train_dataset_dir： 训练数据集路径
use_parallel：      是否开启并行
run_mode：          运行模式，train：训练，finetune：微调，predict：推理
```

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

#### 多机训练

多机多卡微调任务与启动预训练类似，可参考[多机多卡的预训练命令](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/usage/pre_training.html#%E5%A4%9A%E6%9C%BA%E8%AE%AD%E7%BB%83)，并对命令进行如下修改：

1. 增加启动脚本入参`--load_checkpoint /{path}/llama2_7b.ckpt`加载预训练权重。
2. 设置启动脚本中的`--train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord`加载微调数据集。
3. 设置启动脚本中的`--run_mode finetune`，run_mode表示运行模式，train：训练，finetune：微调，predict：推理。

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

