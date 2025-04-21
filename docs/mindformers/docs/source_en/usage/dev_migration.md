# Development Migration

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/dev_migration.md)

This document describes how to develop and build foundation models based on MindSpore Transformers and complete basic adaptation to start the training and inference processes.

## Building a Foundation Model Based on MindSpore Transformers

The basic components of a foundation model in MindSpore Transformers include the configurations, models, and tokenizers for large language models (LLMs). In addition, to use the run_mindformer.py unified script to start the training or inference process, you need to prepare the `YAML` configuration file for training or inference.

### Writing Configurations

A model configuration is an instance that contains all information about a model. The `__init__` methods of all models in MindSpore Transformers receive a model configuration instance as the input parameter. All submodules of the model are initialized based on the information contained in the configuration instance.

MindSpore Transformers provides the [PretrainedConfig](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.PretrainedConfig.html) class, which provides some common configuration methods. The configuration classes of all models should be inherited from the PretrainedConfig class. Developers only need to define all configuration parameters that help build foundation models. Foundation models of the Transformer type have configuration parameters such as `seq_length`, `hidden_size`, `num_layers`, and `num_heads`, and foundation models of the text type have `vocab_size` in addition.

For details, see the configuration class [LlamaConfig](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.LlamaConfig.html) of the Llama model in MindSpore Transformers.

> If your model is similar to a model in the library, you can reuse the same configurations as the model.

### Writing a Model

The MindSpore Transformers foundation model is developed based on the MindSpore framework. Developers only need to pay attention to the implementation of the model network.

MindSpore Transformers provides the [PretrainedModel](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.PreTrainedModel.html) class, which is responsible for storage model configurations and processing the methods of loading and saving models. All model classes must be inherited from the PretrainedModel class, and the model input must be the same. That is, the input parameters of the `construct` method of the model must be the same. For details about the input parameters and meanings, see the Llama model class [LlamaForCausalLM](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.LlamaForCausalLM.html) in MindSpore Transformers. In addition, the model class must implement some abstract methods of the base class, including:

- `prepare_inputs_for_generation`: method for building input for model inference.
- `prepare_inputs_for_predict_layout`: method for building virtual input for the distributed loading model weight.

For specific meanings, refer to the descriptions in [LlamaForCausalLM](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.LlamaForCausalLM.html).

> If your model structure is similar to that of a model in the library, you can reuse the model.

### Writing a Tokenizer (for LLMs)

A tokenizer is used to process input and output of LLMs. It is required in the workflow of LLMs.

MindSpore Transformers provides the [PretrainedTokenizer](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.PreTrainedTokenizer.html) and [PretrainedTokenizerFast](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.PreTrainedTokenizerFast.html) classes, which use Python only and use the Rust library, respectively. The features of the latter one are as follows:

- Faster batch processing.
- Additional methods for mapping between text strings and lexical spaces. For example, the indexes of the lexical element containing a given character or the character spans corresponding to the given lexical element are obtained.

All tokenizer classes must be inherited from the PretrainedTokenizer or PretrainedTokenizerFast class. For details, see [LlamaTokenizer](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.LlamaTokenizer.html) and [LlamaTokenizerFast](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/models/mindformers.models.LlamaTokenizerFast.html).

> If your tokenizer is similar to that in the library, you can reuse that in the library.

### Preparing a Weight and a Dataset

If a PyTorch-based model weight already exists, you can convert the weight to that in the MindSpore format by referring to [Weight Conversion](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/weight_conversion.html).

For details about how to prepare a dataset, see [Dataset](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/dataset.html) or the model document, for example, [Llama2 Description Document > Dataset Preparation](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87).

### Preparing a `YAML` File

MindSpore Transformers uses a `YAML` file to configure all parameters required by a task, including model parameters, training parameters (such as optimizer, learning rate, and dataset), inference parameters (such as tokenizer), distributed parallel parameters, and context environment parameters.

The code of the customized model is not in the MindSpore Transformers library, and the customized module in the code is not registered with MindSpore Transformers. Therefore, the customized model cannot be automatically instantiated. The code is also called external code (for example, the code in the `research` directory). Therefore, you need to add the `auto_register` configuration item for automatically registering any module to the corresponding module configuration in the `YAML` file and set the configuration items to the relative import paths of the API to be registered. When the run_mindformer.py script is executed to start the task, you need to add the input parameter `--register_path` of the registration path and set it to the relative path of the directory where the external code is located.

For example, in the `YAML` file [`research/llama3_1/predict_llama3_1_8b.yaml`](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml) of the Llama3.1-8B model inference in the `research` directory, the configuration item `auto_register` is added for automatic registration to register the customized `Llama3Tokenizer` in [`research/llama3_1/llama3_1_tokenizer.py`](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3_1/llama3_1_tokenizer.py).

```yaml
...
processor:
  return_tensors: ms
  tokenizer:
    model_max_length: 8192
    vocab_file: "/path/tokenizer.model"
    pad_token: "<|reserved_special_token_0|>"
    type: Llama3Tokenizer
    auto_register: llama3_1_tokenizer.Llama3Tokenizer
  type: LlamaProcessor
...
```

The relative import path `auto_register: llama3_1_tokenizer.Llama3Tokenizer` of `Llama3Tokenizer` is configured under `tokenizer`.

Also, `vocab_file` under `tokenizer` should configure as the real path to the tokenizer `tokenizer.model`.

Run the following command to start the inference job:

```bash
python run_mindformer.py --config research/llama3_1/predict_llama3_1_8b.yaml --load_checkpoint path/to/llama3_1_8b.ckpt --register_path research/llama3_1 --predict_data "hello"
```

**Parameters**

|    Parameter    | Description                                               |
|:---------------:|:----------------------------------------------------------|
|     config      | Path of the `YAML` file.                                  |
| load_checkpoint | Loaded weight path.                                       |
|  register_path  | Path of the directory where the external code is located. |
|  predict_data   | Input data for inference.                                 |

`register_path` is set to `research/llama3_1` (path of the directory where the external code is located). For details about how to prepare the model weight, see [Llama3.1 Description Document > Model Weight Download](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3_1/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD).

For details about the configuration file and configurable items, see [Configuration File Descriptions](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/conf_files.html). When compiling a configuration file, you can refer to an existing configuration file in the library, for example, [Llama2-7B fine-tuning configuration file](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/finetune_llama2_7b.yaml).

After all the preceding basic elements are prepared, you can refer to other documents in the MindSpore Transformers tutorial to perform model training, fine-tuning, and inference. For details about subsequent model debugging and optimization, see [Large Model Accuracy Optimization Guide](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/acc_optimize/acc_optimize.html) and [Large Model Performance Optimization Guide](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/perf_optimize/perf_optimize.html).

### Contributing Models to the MindSpore Transformers Open Source Repository

You can contribute models to the MindSpore Transformers open source repository for developers to research and use. For details, see [MindSpore Transformers Contribution Guidelines](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/faq/mindformers_contribution.html).

## MindSpore Transformers Model Migration Practice

### Migration from Llama2-7B to Llama3-8B

Llama3-8B and Llama2-7B have the same model structure but different model parameters, tokenizers, and weights.

#### Model Configurations

The following compares the model configurations between Llama2-7B and Llama3-8B.

![model_config_comparison](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/usage/image/model_config_comparison.png)

The differences are as follows:

- The sequence length of Llama3-8B is 8192. Therefore, `seq_length` is set to `8192`.
- Llama3-8B uses GQA and the number of heads in each key-value group is 8. Therefore, `n_kv_head` is set to `8`.
- The size of the Llama3-8B vocabulary is 128,256. Therefore, `vocab_size` is set to `128256`.
- Llama3-8B expands the hidden layer size of the feed-forward network to 14,336. Therefore, `intermediate_size` is set to `14336`.
- In Llama3-8B, the special word metaindex is modified. Therefore, `bos_token_id` is set to `128000`, `eos_token_id` is set to `128001`, and `pad_token_id` is set to `128002`.
- In Llama3-8B, the value of **theta** in the rotation position code is changed to **500000**. Therefore, `theta` is set to `500000`.

After modifying the corresponding content in the `YAML` file of Llama2-7B, you can obtain the [Llama3-8B configuration file](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_8b/finetune_llama3_8b.yaml).

#### Tokenizer

Llama3-8B re-implements the tokenizer. According to the official implementation, PretrainedTokenizer is inherited from MindSpore Transformers to implement Llama3Tokenizer, which is written in [llama3_tokenizer.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_tokenizer.py).

#### Weight Conversion

The parameters of Llama3-8B are the same as those of Llama2-7B. Therefore, the weight conversion process of Llama2-7B can be reused. For details, see [Llama3 Document > Weight Conversion](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2).

#### Dataset Processing

The tokenizer of Llama3-8B is different from that of Llama2-7B. Therefore, you need to replace the tokenizer of Llama3-8B to preprocess data based on the dataset processing script of Llama2-7B. For details, see [conversation.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_conversation.py) and [llama_preprocess.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/llama3_preprocess.py).

For details about the implementation of Llama3 in MindSpore Transformers, see [Llama3 folder](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/llama3) in the MindSpore Transformers repository. For details about how to use Llama3 in MindSpore Transformers, see [LLama3 documents](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3/README.md).
