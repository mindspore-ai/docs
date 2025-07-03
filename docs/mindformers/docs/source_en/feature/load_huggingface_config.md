# Loading Hugging Face Model Configuration

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/load_huggingface_config.md)

## Overview

Currently, MindSpore Transformers supports loading model configurations from Hugging Face, allowing users to directly load the configuration of models on Hugging Face, while only a few of MindSpore Transformers' own model configurations need to be defined in the YAML file. The benefits of this feature are mainly as follows:

1. Reduced cost of migrating models from Hugging Face. Users can directly reuse the configurations of community models without the need to manually rewrite them.
2. Facilitating consistent reproduction. By using plug-and-play configuration files, it ensures that model hyperparameters (such as the number of layers, number of attention heads, hidden layer size, etc.) remain consistent with the original model.
3. Ecological reuse, facilitating the inheritance of upstream and downstream toolchains. Users can download model configurations and Tokenizers from Hugging Face, and perform inference or deployment using MindSpore Transformers. This also makes it easier to seamlessly integrate with tools that support Hugging Face formats in the future.

## Use Case

- Currently supports reusing Hugging Face model configurations for inference directly.

## Operation Guide

### Preparing Hugging Face Model Configuration

Taking Qwen3 as an example, download the model configuration files (including config.json and generation.json) from the Hugging Face official website and store them in the local folder `./local/qwen3`.

### Preparing YAML Configuration File

This feature only involves the model and inference configurations, with the relevant parameters as follows:

- pretrained_model_dir: The directory path where the Hugging Face model configuration is located;
- model_config: Model configuration fields specific to MindSpore Transformers;
- generation: Parameters related to text generation. Optional configuration, increase if customization is needed. For the configuration items, refer to [GenerationConfig](https://www.mindspore.cn/mindformers/docs/en/dev/generation/mindformers.generation.GenerationConfig.html).

```yaml
pretrained_model_dir: "./local/qwen3"
model:
  model_config:
    compute_dtype: "bfloat16"
    layernorm_compute_dtype: "float32"
    rotary_dtype: "bfloat16"
    params_dtype: "bfloat16"
```

If there is no need to reuse Hugging Face model configurations, MindSpore Transformers requires all necessary fields to be configured in model_config and generation, among which model_type and architectures are required fields.

```yaml
model:
  model_config:
    model_type: qwen3
    architectures: ['Qwen3ForCausalLM']
    ...
    compute_dtype: "bfloat16"
    layernorm_compute_dtype: "float32"
    rotary_dtype: "bfloat16"
    params_dtype: "bfloat16"
generation:
  max_length: 30
  ...
```

> The configuration fields for the model in the YAML file take precedence over the corresponding model configurations in pretrained_model_dir. Therefore, if there are fields with the same name, the fields in the YAML file will override the original values.

### Initiating Tasks

Refer to [Using run_mindformer.py to initiate inference tasks](https://www.mindspore.cn/mindformers/docs/en/dev/guide/inference.html#using-run-mindformer-once-to-start-the-inference-script).

## Frequently Asked Questions

- If Hugging Face model configuration is not loaded, and model_type and architectures are required configuration fields, how should it be configured?

    Taking Qwen3 as an example:

    If its model configuration class Qwen3Config is registered with non-empty search_names, then model_type only needs to be configured with the value of search_names; If search_names is not provided, then model_type should be configured as Qwen3Config; architectures should be configured as the name of the corresponding model class Qwen3ForCausalLM.
