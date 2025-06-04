# SFT微调

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/supervised_fine_tuning.md)

## 概述

SFT（Supervised Fine-Tuning，监督微调）采用有监督学习思想，是指在预训练模型的基础上，通过调整部分或全部参数，使其更适应特定任务或数据集的过程。

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

## SFT微调方式

MindSpore Transformers当前支持全参微调和LoRA低参微调两种SFT微调方式。全参微调是指在训练过程中对所有参数进行更新，适用于大规模数据精调，能获得最优的任务适应能力，但需要的计算资源较大。LoRA低参微调在训练过程中仅更新部分参数，相比全参微调显存占用更少、训练速度更快，但在某些任务中的效果不如全参微调。

### LoRA 原理简介

LoRA通过将原始模型的权重矩阵分解为两个低秩矩阵来实现参数量的显著减少。例如，假设一个权重矩阵W的大小为m x n，通过LoRA，该矩阵被分解为两个低秩矩阵A和B，其中A的大小为m x r，B的大小为r x n（r远小于m和n）。在微调过程中，仅对这两个低秩矩阵进行更新，而不改变原始模型的其他部分。

这种方法不仅大幅度降低了微调的计算开销，还保留了模型的原始性能，特别适用于数据量有限、计算资源受限的环境中进行模型优化，详细原理可以查看论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 。

## 使用MindSpore Transformers进行全参微调

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

MindSpore Transformers提供权重转换脚本，通过执行[convert_weight.py转换脚本](https://gitee.com/mindspore/mindformers/blob/dev/convert_weight.py)，可以将HuggingFace的权重转换为完整的ckpt权重。

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

1. 执行MindSpore Transformers中的[alpaca_converter.py脚本](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py)，将数据集转换为多轮对话格式。

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

2. 执行MindSpore Transformers中的[llama_preprocess.py脚本](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py)，将数据转换为MindRecord格式。该操作依赖fastchat工具包解析prompt模板, 请提前安装fastchat >= 0.2.13。

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

#### 单卡训练

执行`run_mindformer.py`启动单卡的微调任务，下面提供了一个使用示例：

以Llama2模型单卡微调为例，由于单卡显存有限，无法运行完整的Llama2-7B模型，所以缩层进行示例，修改`finetune_llama2_7b.yaml`，将其中`num_layers`设置为2。

启动命令如下：

```shell
python run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --use_parallel False \
 --run_mode finetune
```

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

多机多卡微调任务与启动预训练类似，可参考[多机多卡的预训练命令](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html#%E5%A4%9A%E6%9C%BA%E8%AE%AD%E7%BB%83)，并对命令进行如下修改：

1. 增加启动脚本入参`--load_checkpoint /{path}/llama2_7b.ckpt`加载预训练权重。
2. 设置启动脚本中的`--train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord`加载微调数据集。
3. 设置启动脚本中的`--run_mode finetune`，run_mode表示运行模式，train：训练，finetune：微调，predict：推理。

任务执行完成后，在mindformers/output目录下，会生成checkpoint文件夹，同时模型文件会保存在该文件夹下。

## 使用MindSpore Transformers进行LoRA低参微调

MindSpore Transformers支持配置化使能LoRA微调，无需对每个模型进行代码适配，而仅需修改全参微调的YAML配置文件中的模型配置，添加 `pet_config` 低参微调配置，即可使用其进行LoRA低参微调任务。以下展示了Llama2模型LoRA微调的YAML配置文件中的模型配置部分，并对 `pet_config` 参数进行了详细说明。

### 示例配置文件（YAML）

完整的YAML配置文件可以通过以下链接访问：[Llama2 LoRA微调 YAML 文件](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/lora_llama2_7b.yaml)。

```yaml
# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1
    seq_length: 4096
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    vocab_size: 32000
    compute_dtype: "float16"
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
  arch:
    type: LlamaForCausalLM
```

### pet_config 参数详解

在 model_config 中，pet_config 是LoRA微调的核心配置部分，用于指定LoRA的相关参数。具体参数说明如下：

- **pet_type:** 指定参数高效微调技术（PET，Parameter-Efficient Tuning）的类型为LoRA。这意味着在模型的关键层中会插入LoRA模块，以减少微调时所需的参数量。
- **lora_rank:** 定义了低秩矩阵的秩值。秩值越小，微调时需要更新的参数越少，从而减少计算资源的占用。这里设为16是一个常见的平衡点，在保持模型性能的同时，显著减少了参数量。
- **lora_alpha:** 控制LoRA模块中权重更新的缩放比例。这个值决定了微调过程中，权重更新的幅度和影响程度。设为16表示缩放幅度适中，有助于稳定训练过程。
- **lora_dropout:** 设置LoRA模块中的dropout概率。Dropout是一种正则化技术，用于减少过拟合风险。设置为0.05表示在训练过程中有5%的概率会随机“关闭”某些神经元连接，这在数据量有限的情况下尤为重要。
- **target_modules:** 通过正则表达式指定LoRA将应用于模型中的哪些权重矩阵。在Llama中，这里的配置将LoRA应用于模型的自注意力机制中的Query（wq）、Key（wk）、Value（wv）和Output（wo）矩阵。这些矩阵在Transformer结构中扮演关键角色，插入LoRA后可以在减少参数量的同时保持模型性能。

### Llama2-7B 的 LoRA 微调示例

MindSpore Transformers 提供了 Llama2-7B 的 [LoRA 微调示例](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#lora%E5%BE%AE%E8%B0%83)。微调过程中使用的数据集可以参考[数据集下载](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)获得。

以 Llama2-7B 为例，可以执行以下 msrun 启动脚本，进行 8 卡分布式微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" 8
```

当权重的分布式策略和模型的分布式策略不一致时，需要对权重进行切分转换。加载权重路径应设置为以 `rank_0` 命名的目录的上一层路径，同时开启权重自动切分转换功能 `--auto_trans_ckpt True` 。关于分布式权重切分转换的场景和使用方式的更多说明请参考[分布式权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/transform_weight.html)。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/checkpoint/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```
