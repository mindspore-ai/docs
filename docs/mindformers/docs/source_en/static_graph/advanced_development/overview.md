<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Advanced Development Overview

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/advanced_development/overview.md)

MindSpore Transformers advanced development targets scenarios such as model migration, tuning, and accuracy verification, helping users go beyond basic training and inference to perform development migration, debugging, optimization, and accuracy comparison. This section summarizes all advanced development documentation by category: **Diagnostics and Optimization**, **Model Development and Configuration**, **Accuracy Comparison**, and **API Reference**, for quick reference and navigation.

## Diagnostics and Optimization

We provide systematic methods for identifying and resolving precision and performance issues during training and inference.

| Document                                                                                                                           | Description                                                                                                                                              | Architecture Support |
|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| [Precision Optimization](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/precision_optimization.html)     | Common precision issues in large model training and general troubleshooting methods, including checklists, parameter alignment, and long-run validation. | Mcore/Legacy         |
| [Performance Optimization](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/performance_optimization.html) | Large model performance tuning approach and tools, covering data loading, forward/backward computation, communication, scheduling, and Profile usage.    | Mcore/Legacy         |

## Model Development and Configuration

Support for building or migrating models from scratch and quickly launching training and inference via configuration templates.

| Document                                                                                                                                                   | Description                                                                                                                                           | Architecture Support |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| [Development Migration](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/dev_migration.html)                                       | End-to-end workflow for building large models with MindSpore Transformers, including config, model, tokenizer, and YAML configuration.                | Legacy               |
| [Guide to Using the Inference Configuration Template](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/yaml_config_inference.html) | How to use the YAML configuration template for inference, including quick setup with Hugging Face/ModelScope model directories.                       | Mcore                |
| [Training Template Instruction](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/training_template_instruction.html)               | Overview of general configuration templates for pre-training and fine-tuning (dense/MoE, etc.) and quick start for custom or unsupported model sizes. | Mcore                |
| [Weight Transfer](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/weight_transfer.html)                                           | Adapting new models to MindSpore Transformers weight conversion for unified Hugging Face to MindSpore weight conversion and loading.                  | Mcore                |

## Accuracy Comparison

Validation of alignment with reference implementations or GPU environments for both training and inference.

| Document                                                                                                                                          | Description                                                                                                                                           | Architecture Support |
|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| [Compare Training Accuracy with Megatron-LM](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/accuracy_comparison.html)   | Training accuracy alignment with Megatron-LM at the model level, including equivalent structure setup and comparison of forward, loss, and gradients. | Mcore                |
| [Comparison of Inference Precision](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/inference_precision_comparison.html) | Inference precision acceptance workflow and troubleshooting, including online inference checks, dataset evaluation, and common issue resolution.      | Mcore                |

## API Reference

Entry point to API documentation for MindSpore Transformers modules.

| Document                                                                                 | Description                                                                               |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [API](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/api.html) | API index and detailed interface documentation for MindSpore Transformers and submodules. |
