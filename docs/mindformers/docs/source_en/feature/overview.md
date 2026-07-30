# Function Overview

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/feature/overview.md)

MindSpore Transformers dynamic graph (PyNative) provides various functions for the entire training process of **pre-training/fine-tuning**, facilitating configuration-based development and optimization. This section summarizes the functions by **general functions** and **training functions**. The inference and deployment capabilities are currently provided by static graphs and are listed in "Static Graph Features."

## General Functions

| Function    | Description                             |
|--------|---------------------------------|
| Task startup  | One-click startup of single-device, single-node multi-device, and multi-node tasks based on `msrun`.  |
| Configuration file description| YAML in dataclass style, which centrally manages all configurable items for training.|
| Logs    | Log structure and storage description.                     |

## Training Functions

| Function               | Description                                                                          |
|-------------------|------------------------------------------------------------------------------|
| Dataset              | Megatron dataset (`BlendedMegatronDatasetDataLoader`) with preprocessed `.bin`/`.idx`, supporting multi-source mixing.|
| Hyperparameters and optimizers for training        | AdamW/Muon optimizer and learning rate strategy with warmup.                                           |
| Distributed parallel training          | DP/FSDP, TP, PP, CP, EP, and SP multi-dimensional hybrid parallelism.                                              |
| Training memory optimization           | Recomputing (full/select), fine-grained SWAP, and CPU offload.                                      |
| Safetensors weight   | Safetensors shard saving and loading, supporting asynchronous saving and redundancy elimination.                                            |
| Resumable training             | Step-level resumable training, reducing the loss caused by interruptions in large-scale training.                                                     |
| Training metric monitoring and profiling| grad/param norm, loss monitoring, and MaxLogits value health monitoring and performance analysis.                                |
| Other training features           | Gradient accumulation, gradient clipping, and operator fusion.                                                            |

## Static Graph Features

For details about the capabilities that are not covered by dynamic graphs, such as inference and deployment, see [Static Graph Features](./static_graph_features.md).
