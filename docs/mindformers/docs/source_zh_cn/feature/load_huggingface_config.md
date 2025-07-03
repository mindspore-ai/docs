# 加载 Hugging Face 模型配置

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/load_huggingface_config.md)

## 概述

当前 MindSpore Transformers 已支持加载 Hugging Face 的模型配置，用户可以直接加载 Hugging Face 上模型的配置，而 yaml 中只需要定义少数 MindSpore Transformers 自有的模型配置。本特性带来的好处主要如下：

1. 降低从 Hugging Face 迁移模型的成本。用户可以直接复用社区模型的配置，而无需手动重写。
2. 便于复现一致性。通过即插即用配置文件，保证了模型超参数（如层数、注意力头数、隐藏层大小等）与原模型保持一致。
3. 生态复用，方便继承上下游工具链。用户可以在 Hugging Face 上下载模型配置和 Tokenizer，使用 MindSpore Transformers 进行推理或部署。也便于后续与支持 Hugging Face 格式的工具无缝对接。

## 使用场景

- 当前支持复用 Hugging Face 模型配置直接进行推理。

## 操作指南

### 准备 Hugging Face 模型配置

以 Qwen3 为例，从 Hugging Face 官网下载模型的配置文件（包括 config.json和generation.json）存放在本地文件夹`./local/qwen3`。

### 准备 yaml 配置文件

该特性只涉及模型和推理配置，相关参数如下：

- pretrained_model_dir：Hugging Face 模型配置所在的目录路径；
- model_config：MindSpore Transformers 自有的模型配置字段；
- generation：文本生成相关的参数。可选配置，如需自定义则增加。其下的配置项可以参考[GenerationConfig](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/generation/mindformers.generation.GenerationConfig.html)。

```yaml
pretrained_model_dir: "./local/qwen3"
model:
  model_config:
    compute_dtype: "bfloat16"
    layernorm_compute_dtype: "float32"
    rotary_dtype: "bfloat16"
    params_dtype: "bfloat16"
```

若不需要复用 Hugging Face 模型配置，MindSpore Transformers 需要在 model_config 和 generation 配置所有所需字段，其中 model_type 和 architectures 为必须配置字段。

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

> yaml 中模型配置字段优先级大于 pretrained_model_dir 中对应模型配置，因此存在相同配置字段时，yaml 中的字段会覆盖掉原有值。

### 拉起任务

参考[使用run_mindformer.py启动推理任务](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/inference.html#%E4%BD%BF%E7%94%A8-run-mindformer-%E4%B8%80%E9%94%AE%E5%90%AF%E5%8A%A8%E8%84%9A%E6%9C%AC%E6%8E%A8%E7%90%86)。

## 常见问题FAQ

- 若不加载 Hugging Face 模型配置， model_type 和 architectures 为必须配置字段，该如何配置？

    以 Qwen3 为例：

    注册其模型配置类 Qwen3Config 时，若传入参数 search_names 非空，则 model_type 只需要配置为 search_names 的值即可；若未传入参数 search_names，则 model_type 配置成 Qwen3Config 即可；architectures 配置成对应的模型类名称 Qwen3ForCausalLM 即可。
