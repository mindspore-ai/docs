# 使用Tokenizer

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/tokenizer.md)

## 概述

Hugging Face Tokenizer 是由 Hugging Face 开发的一款高效、灵活的文本分词工具。它旨在为自然语言处理（NLP）任务提供强大的支持，通过将文本转换为模型能够理解的形式——即分词（tokens）。Tokenizer 不仅负责将文本分割成词汇单元，还管理着这些词汇单元与它们对应的索引之间的映射关系，这在机器学习模型中用于输入表示至关重要。

MindSpore Transformers中涉及使用Tokenizer的流程有：推理、微调在线数据集加载及离线数据集预处理等。当前已支持直接使用基于Hugging Face transformers的Tokenizer。

MindSpore Transformers原有的Tokenizer组件与Hugging Face Tokenizer的功能相同，直接使用无需额外开发成本，对于迁移Hugging Face上的模型时比较友好。本文档主要介绍以推理流程为例，介绍如何复用Hugging Face Tokenizer。目前仅支持新架构的Qwen3系列模型，后续持续优化泛化性。

## 基本流程

使用流程可以分解成以下几个步骤：

### 1. 根据模型选择下载Tokenizer文件

根据模型下载对应的Tokenizer相关文件到对应的文件夹，文件包括词表文件等。此外，Hugging Face的Tokenizer具体可以分为两大类：

1. transformers的内置Tokenizer。如Qwen2Tokenizer；

2. 继承transformers的Tokenizer的基类实现的自定义的Tokenizer，并没有合入transformers。只是在Hugging Face的仓库上或者本地存在Tokenizer的Python文件，需要支持远程加载和将Tokenizer的Python文件存到对应文件夹。如ChatGLM4Tokenizer。

### 2. 修改配置文件

根据任务参考后面的[推理流程示例](#推理流程示例)和[训练流程示例](#训练流程示例)修改配置文件。

### 3. 执行任务

参考样例拉起任务。

## 推理流程示例

推理流程以Qwen3模型为例。

### 使用run_mindformer.py脚本启动

1. 修改yaml配置

    Qwen3模型的配置文件[predict_qwen3.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/qwen3/predict_qwen3.yaml)需要修改的地方如下：

    ```yaml
    use_legacy: False

    pretrained_model_dir: "path/to/qwen3_dir"
    ```

    参数说明：

    - use_legacy：决定是否使用老架构，默认值：`True`；
    - pretrained_model_dir：放置Tokenizer相关文件的文件夹路径。

2. 拉起任务

    以`Qwen3-8b`的单卡推理为例，启动命令如下：

    ```shell
    python run_mindformer.py \
    --config configs/qwen3/predict_qwen3.yaml \
    --load_checkpoint /path/to/model_dir \
    --run_mode predict \
    --trust_remote_code False \
    --predict_data '帮助我制定一份去上海的旅游攻略'
    ```

    参数说明：

    - config：yaml配置文件的路径；
    - load_checkpoint：放置权重的文件夹路径；
    - run_mode：运行模式，推理任务配置为`predict`；
    - trust_remote_code：是否信任从远程下载的代码，默认值：`False`;
    - predict_data：推理的输入。

### 自定义脚本

推理的自定义脚本实现过程涉及Tokenizer的实例化，其实现代码参考如下：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="path/to/pretrained_model_dir",
                                          trust_remote_code=False)
```

参数说明：

- pretrained_model_name_or_path：Hugging Face下载的Tokenizer相关的文件存储的文件夹路径。
- trust_remote_code：是否信任从远程下载的代码，默认值：`False`。

## 训练流程示例

### 在线数据集加载

修改yaml配置中train_dataset部分中和Tokenizer相关的部分。

```yaml
use_legacy: False

pretrained_model_dir: &pretrained_model_dir "path/to/qwen3_dir"

train_dataset: &train_dataset
    data_loader:
        type: CommonDataLoader
        handler:
            - type: AlpacaInsturctDataHandler
            pretrained_model_dir: *pretrained_model_dir
            trust_remote_code: False
            tokenizer:
                padding_side: "right"
```

参数说明：

- use_legacy：决定是否使用老架构，默认值：`True`；
- pretrained_model_dir：HuggingFace下载的Tokenizer相关的文件存储的文件夹路径。
- padding_side: 指定Tokenizer的padding的位置，训练时需要设置：`"right"`。
- trust_remote_code：是否信任从远程下载的代码，默认值：`False`。

### 离线数据集预处理

将离线数据集预处理的脚本中Tokenizer实例化的代码替换成以下代码即可：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="path/to/pretrained_model_dir",
                                          trust_remote_code=False)
tokenizer.padding_side = "right"
```

参数说明：

- pretrained_model_name_or_path：HuggingFace下载的Tokenizer相关的文件存储的文件夹路径。
- trust_remote_code：是否信任从远程下载的代码，默认值：`False`。

关于Tokenizer的支持的更多功能参考Tokenizer的[API接口文档](https://hf-mirror.com/docs/transformers/main_classes/tokenizer)，使用方法可以参考其[使用文档](https://hf-mirror.com/docs/transformers/main/en/fast_tokenizers)。