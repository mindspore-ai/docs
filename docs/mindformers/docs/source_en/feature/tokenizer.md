# Using Tokenizer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/tokenizer.md)

## Overview

Hugging Face Tokenizer is an efficient and flexible text word segmentation tool developed by Hugging Face. It aims to provide strong support for natural language processing (NLP) tasks by converting text into a form that the model can understand - that is, tokens. Tokenizer is not only responsible for dividing text into lexical units, but also manages the mapping relationship between these lexical units and their corresponding indexes, which is crucial for input representation in machine learning models.

The processes involving the use of Tokenizer in MindSpore Transformers include: inference, online dataset loading when fine-tuning, and preprocessing of offline datasets, etc. Currently, direct use of Tokenizers based on Hugging Face transformers is supported.

The original Tokenizer component of MindSpore Transformers has the same function as the Hugging Face Tokenizer. It can be used directly without additional development costs and is relatively friendly when migrating models on Hugging Face. This document mainly introduces how to reuse Hugging Face Tokenizer by taking the reasoning process as an example. Currently, only the Qwen3 series models of the new architecture are supported, and the generalization ability will be continuously optimized in the future.

## Basic Process

The usage process can be decomposed into the following steps:

### 1. Select and Download the Tokenizer File Based on the Model

Download the corresponding Tokenizer-related files to the corresponding folder based on the model. The files include word list files, etc. Furthermore, Hugging Face's tokenizers can be specifically divided into two major categories:

1. The built-in Tokenizer of transformers. For example, Qwen2Tokenizer;

2. A custom Tokenizer implemented by inheriting the base class of the Tokenizer from transformers is not merged into transformers. Only the Python files of the Tokenizer exist on Hugging Face's repository or locally. It is necessary to support remote loading and saving the Python files of the Tokenizer to the corresponding folders. Such as ChatGLM4Tokenizer.

### 2. Modify the Configuration File

Modify the configuration file according to the [Inference Process Example](#inference-process-example) and [Training Process Example](#training-process-example) following the task reference.

### 3. Carry Out the Task

Refer to the sample to start the task.

## Inference Process Example

The inference process takes the Qwen3 model as an example.

### Start Using the run_mindformer.py Script

1. Modify the yaml configuration

    Qwen3 model configuration file [predict_qwen3 yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/qwen3/predict_qwen3.yaml) needs to be modified The places are as follows:

    ```yaml
    use_legacy: False

    pretrained_model_dir: "path/to/qwen3_dir"
    ```

    Parameter description:

    - use_legacy: Decide whether to use the old architecture, default value: `True`;
    - pretrained_model_dir: The folder path where Tokenizer-related files are placed.

2. Pull up the task

    Taking the single-card inference of Qwen3-8b as an example, the startup command is as follows:

    ```shell
    python run_mindformer.py \
    --config configs/qwen3/predict_qwen3.yaml \
    --load_checkpoint /path/to/model_dir \
    --run_mode predict \
    --trust_remote_code False \
    --predict_data '帮助我制定一份去上海的旅游攻略'
    ```

    Parameter description:

    - config: The path of the yaml configuration file.
    - load_checkpoint: The folder path where the weights are placed.
    - run_mode: Operation mode, the inference task is configured as `predict`.
    - trust_remote_code: Whether to trust the code downloaded remotely, default value: `False`.
    - predict_data: Input for reasoning.

### Custom Script

The custom script implementation process of reasoning involves the instantiation of the Tokenizer, and its implementation code is as follows:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="path/to/pretrained_model_dir",
                                          trust_remote_code=False)
```

Parameter description:

- pretrained_model_name_or_path: The folder path where the files related to the Tokenizer downloaded by HuggingFace are stored.
- trust_remote_code: Whether to trust the code downloaded remotely, default value: 'False'.

## Training Process Example

### Online Dataset Loading

Modify the part related to Tokenizer in the train_dataset section of the yaml configuration:

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

Parameter description

- use_legacy: Decide whether to use the old architecture, default value: `True`.
- pretrained_model_dir: The folder path where the files related to the Tokenizer downloaded by HuggingFace are stored.
- padding_side: Specifies the padding position of the Tokenizer. During training, it needs to be set as: `"right"`.
- trust_remote_code: Whether to trust the code downloaded remotely, default value: `False`.

### Preprocessing of Offline Datasets

Just replace the code for instantiating the Tokenizer in the script for preprocessing the offline dataset with the following code:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="path/to/pretrained_model_dir",
                                          trust_remote_code=False)
tokenizer.padding_side = "right"
```

Parameter description:

- pretrained_model_name_or_path: The folder path where the files related to the Tokenizer downloaded by HuggingFace are stored.
- trust_remote_code: Whether to trust the code downloaded remotely, default value: `False`.

For more features supported by Tokenizer, refer to [API interface document](https://hf-mirror.com/docs/transformers/main_classes/tokenizer), using method can refer to the [using document](https://hf-mirror.com/docs/transformers/main/en/fast_tokenizers).