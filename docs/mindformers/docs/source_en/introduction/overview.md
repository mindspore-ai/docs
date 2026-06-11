# Overall Structure

Starting with **r2.0.0**, MindSpore Transformers has adopted a **dynamic graph (PyNative) implementation** as its primary development path. This chapter introduces the overall architecture, core modules, and training capabilities of the dynamic graph training stack, and provides a minimal starting point for implementation.

```{admonition} The Limits of Dynamic Graph Capabilities
:class: note

- The source code for the dynamic graph is located in `mindformers/pynative/`.
- The current dynamic graph focuses on **pre-training and fine-tuning** scenarios; capabilities such as inference, service deployment, and quantization are still provided by the static graph. For more details, see the [Static Graph Implementation](../static_graph/introduction/overview.md) section.
```

---

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/introduction/overview.md)

## Overview

The dynamic graph implementation adopts a **layered, modular** design:

- **Entry Layer**: Unified script `run_mindformer.py`, routes to dynamic graph trainer via `--mode 1` (the `config.mode == 1` branch in the source code of `run_mindformer.py`).
- **Control Layer**: `mindformers.pynative.trainer.Trainer` is responsible for building the model/dataset/optimizer, and drives the training loop.
- **Configuration Layer**: Centralized management of YAML configurations using dataclasses, with parsing and validation completed during loading.
- **Capability Layer**: Implements multi-dimensional parallelism, fused operators, and memory optimization based on MindSpore's dynamic graph capabilities.

First, look at the "Execution Flowchart", then the "Module Layering Diagram".

### Execution Flow

```text
run_mindformer.py --mode 1            # Entry, PYNATIVE_MODE routing
        │
        ▼
mindformers.pynative.trainer.Trainer  # Builds model/dataset/optimizer, drives training loop
        │
        ├── config/        YAML → dataclass configuration system
        ├── base_models/   GPTModel (Unified interface for Dense/MoE)
        ├── distributed/   Multi-dimensional parallelism and memory optimization
        ├── optimizer/     AdamW/Muon
        ├── loss/          Fused cross-entropy
        ├── callback/      Weight saving, Loss/Metric monitoring
        └── tools/         Monitoring & Profiling
```

### Module Layering

The model side is divided into three layers from top to bottom: "Model → Transformer Components → Primitive Layer", corresponding one-to-one with the core module table below:

```text
base_models/gpt/  GPTModel (Unified interface for Dense/MoE, assembled via ModuleSpec)
        │
        ▼
transformers/     Attention · MLA · MTP · TransformerLayer/Block · MLP · MoE
        │                                   (router/experts/shared_experts)
        ▼
layers/           Linear · RMSNorm · SwiGlu · FlashAttention · Mask generation
```

> `base_models/common/embeddings` provides positional encodings such as RoPE, YaRN, etc., for use by the above layers.

---

## Core Modules

The sub-modules of the dynamic graph implementation and their responsibilities are as follows:

| Module | Path | Responsibility |
|------|------|------|
| Trainer | `pynative/trainer/` | Training controller `Trainer`: Builds model/data/optimizer, executes forward/backward, gradient synchronization, saving, and state tracking. |
| Configuration | `pynative/config/` | Centralized configuration management via dataclasses, supports loading from YAML and validation. |
| Distributed | `pynative/distributed/` | Device mesh construction and sharding for multi-dimensional parallelism, as well as various memory optimizations. |
| Optimizer | `pynative/optimizer/` | `AdamW` and `Muon` optimizer implementations, supporting distributed synchronization and mixed precision. |
| Loss | `pynative/loss/` | Dynamic graph fused cross-entropy `CrossEntropyLoss`, implemented via custom `_LogSoftmax` + `_NLLLoss` (with manual backward). |
| Base Model | `pynative/base_models/gpt/` | General `GPTModel`, uniformly supports Dense and MoE, uses `ModuleSpec` mechanism to build models according to configuration; `common/embeddings` provides RoPE, YaRN and other positional encodings. |
| Transformer Components | `pynative/transformers/` | Attention, MLA, MTP, TransformerLayer/Block, MLP, and MoE sub-modules. |
| Primitive Layer | `pynative/layers/` | Fused operators such as Linear, LayerNorm/RMSNorm, SwiGlu, Flash Attention, mask generation, etc. |
| Callback | `pynative/callback/` | Checkpoint saving, Loss monitoring, training metric monitoring, MaxLogits health check. |
| Tools | `pynative/tools/` | Metric monitoring aggregation (`MonitorGroup`) and Profiling. |

The following sections expand on the "Configuration" and "Distributed" modules, which contain more detailed information.

### Configuration Module Dataclasses

`pynative/config/` maps each section of YAML into independent dataclasses, facilitating validation and default value management. Commonly used configuration classes:

| dataclass | Corresponding Responsibility |
|---|---|
| `CheckpointConfig` | Weight saving/loading |
| `TrainingConfig` | Training steps, batch size, gradient accumulation, etc. |
| `ParallelismConfig` | Multi-dimensional parallelism dimensions |
| `OptimizerConfig` | Optimizer type and hyperparameters |
| `LrSchedulerConfig` | Learning rate strategy |
| `ModelConfig` | Model structure parameters |
| `MonitorConfig` | Metric monitoring and visualization |

### Distributed Module Capabilities

`pynative/distributed/` undertakes both "parallelism sharding" and "memory optimization" responsibilities:

- **Parallelism Dimensions**: DP (including FSDP/HSDP parameter sharding), TP, PP, CP, EP, SP. The device mesh is constructed based on the product of each dimension, satisfying `dp_replicate * dp_shard * cp * tp * pp == world_size` (`parallel_dims.py`).
- **Memory Optimization**: Activation checkpointing, fine-grained SWAP, CPU offload.

```{admonition} About pet (LoRA) and models subdirectories
:class: warning

The `pynative/pet/` and `pynative/models/` directories currently only contain `__init__.py` and have no implementation yet. LoRA fine-tuning is **not yet implemented** in the dynamic graph: when triggered, it will raise `NotImplementedError("Lora model is not implemented yet.")` in `trainer/utils.py`. For LoRA, please use the static graph implementation.
```

---

## Model Architecture

The dynamic graph adopts a **hierarchical abstraction + modular** design: `GPTModel` (General PreTrained Model) serves as the unified model interface, composing modular interfaces downwards such as `TransformerBlock`, `MoELayer`, `Attention`, `Linear`, `Embedding`, `Norm`, etc., and freely combines them to build models through the `ModuleSpec` mechanism. All modules have undergone parallel and operator fusion optimizations based on MindSpore's dynamic graph.

Models currently implemented in the dynamic graph include DeepSeek-V3 (MoE + MLA + MTP) and Qwen3 (Dense).

---

## Training Capabilities

The dynamic graph training stack provides the following capabilities (configuration instructions for each capability will be supplemented in subsequent documentation):

- **Multi-dimensional Hybrid Parallelism**: Flexible combination of data parallelism (including FSDP/HSDP parameter sharding), tensor parallelism (TP), pipeline parallelism (PP, supporting 1F1B and interleave), context parallelism (CP, Colossal method), expert parallelism (EP), and sequence parallelism (SP).
- **Optimizers and Learning Rates**: AdamW, Muon; multiple learning rate strategies with warmup.
- **Dataset**: Megatron blended multi-source dataset (`BlendedMegatronDatasetDataLoader`, preprocessed `.bin`/`.idx` files).
- **Memory Optimization**: Activation checkpointing (full/selective), fine-grained SWAP, CPU offload.
- **Checkpoints**: Sharded saving and loading in Safetensors format, supporting asynchronous saving and redundancy elimination.
- **Stability and Observability**: Resuming training from checkpoints, gradient/parameter norm and Loss monitoring, MaxLogits numerical health checks, and Profiling.

---

## Next Steps

After reading the architecture, the minimal path to getting started is as follows.

**Single-card** (for debugging/validation):

```bash
python run_mindformer.py --config <your_config.yaml> --mode 1
```

**Multi-card msrun launch** (actual training, taking 8 cards as an example):

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config <your_config.yaml> --mode 1"
```

`--mode 1` routes to the dynamic graph trainer. The complete "prepare configuration → launch → view results" process and end-to-end training configurations will be supplemented in subsequent documentation (Quick Start, Training Guide, feature-specific pages).

---

## Related Documentation

- Capabilities provided by the static graph (inference/quantization, etc.): [Static Graph Implementation](../static_graph/introduction/overview.md)

> The Installation Guide, Quick Start, Training Guide, and feature-specific pages are being supplemented and will be available in subsequent submissions.