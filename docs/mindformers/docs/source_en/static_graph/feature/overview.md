<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Features Overview

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/feature/overview.md)

MindSpore Transformers provides a wide range of features across the full process of pre-training, fine-tuning, inference, and deployment, enabling configurable development and optimization. This section summarizes all features by category: **General Features**, **Training Features**, and **Inference Features**, for quick reference and navigation.

## General Features

Foundational capabilities reusable across pre-training, fine-tuning, and inference for consistent setup and reuse.

| Feature                                                                                                                              | Description                                                                               | Architecture Support |
|--------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|----------------------|
| [Start Tasks](https://www.mindspore.cn/mindformers/docs/en/master/feature/start_tasks.html)                                          | One-click start for single-device, single-node and multi-node tasks.                      | Mcore/Legacy         |
| [Ckpt Weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/ckpt.html)                                                | [Checkpoint 1.0] Supports conversion, slicing and merging of weight files in ckpt format. | Legacy               |
| [Safetensors Weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html)                                  | [Checkpoint 1.0] Supports saving and loading weight files in safetensors format.          | Mcore/Legacy         |
| [Configuration File Descriptions](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html)                    | Use YAML files to centrally manage and adjust configurable items in tasks.                | Mcore/Legacy         |
| [Loading Hugging Face Model Configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/load_huggingface_config.html) | Plug-and-play loading of Hugging Face community model configurations.                     | Mcore                |
| [Logs](https://www.mindspore.cn/mindformers/docs/en/master/feature/logging.html)                                                     | Introduction to logs, including log structure and log saving.                             | Mcore/Legacy         |
| [Using Tokenizer](https://www.mindspore.cn/mindformers/docs/en/master/feature/tokenizer.html)                                        | Introduction to tokenizer; supports Hugging Face Tokenizer in inference and datasets.     | Mcore                |

## Training Features

Supports large-scale, reliable large model training and tuning.

| Feature                                                                                                                               | Description                                                                                    | Architecture Support |
|---------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------|
| [Dataset](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html)                                                   | Supports multiple types and formats of datasets (Megatron, Hugging Face, MindRecord, etc.).    | Mcore/Legacy         |
| [Training Hyperparameters](https://www.mindspore.cn/mindformers/docs/en/master/feature/training_hyperparameters.html)                 | Flexibly configure hyperparameters (learning rate, optimizer, etc.) for large model training.  | Mcore/Legacy         |
| [Training Metrics Monitoring](https://www.mindspore.cn/mindformers/docs/en/master/feature/monitor.html)                               | Visualization for the training phase to monitor and analyze metrics and information.           | Mcore/Legacy         |
| [Resumable Training After Breakpoint](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training.html)               | [Checkpoint 1.0] Step-level resumable training to reduce waste from unexpected interruptions.  | Mcore/Legacy         |
| [Checkpoint Saving and Loading](https://www.mindspore.cn/mindformers/docs/en/master/feature/checkpoint_saving_and_loading.html)       | [Checkpoint 2.0] Checkpoint saving and loading.                                                | Mcore                |
| [Resumable Training After Breakpoint 2.0](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training2.0.html)        | [Checkpoint 2.0] Step-level resumable training with scaling and incremental scenarios.         | Mcore                |
| [Training High-Availability (Beta)](https://www.mindspore.cn/mindformers/docs/en/master/feature/high_availability.html)               | End-of-life CKPT, UCE fault-tolerant recovery, and process-level rescheduling recovery.        | Mcore                |
| [Distributed Parallel Training](https://www.mindspore.cn/mindformers/docs/en/master/feature/parallel_training.html)                   | One-click multi-dimensional hybrid distributed parallel for efficient training at scale.       | Mcore/Legacy         |
| [Training Memory Optimization](https://www.mindspore.cn/mindformers/docs/en/master/feature/memory_optimization.html)                  | Recomputation and fine-grained activation SWAP to reduce peak memory.                          | Mcore/Legacy         |
| [Data Skip and Health Monitoring](https://www.mindspore.cn/mindformers/docs/en/master/feature/skip_data_and_ckpt_health_monitor.html) | Data skip and checkpoint health monitoring for more robust training.                           | Mcore/Legacy         |
| [Pre-trained Model Average (PMA) Weight Merge](https://www.mindspore.cn/mindformers/docs/en/master/feature/pma_fused_checkpoint.html) | Merge multiple checkpoints (PMA) and fused checkpoint saving.                                  | Mcore                |
| [Other Training Features](https://www.mindspore.cn/mindformers/docs/en/master/feature/other_training_features.html)                   | Gradient accumulation, gradient clipping, CPU affinity, MoE DropRate, RoPE/SwiGLU fusion, etc. | Mcore/Legacy         |

## Inference Features

Targets inference and deployment scenarios, enabling trained models to be deployed efficiently for production use.

| Feature                                                                                       | Description                                                                      | Architecture Support |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------|
| [Quantization](https://www.mindspore.cn/mindformers/docs/en/master/feature/quantization.html) | Integrates MindSpore Golden Stick for a unified quantization inference workflow. | Mcore/Legacy         |
