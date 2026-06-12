# 整体架构

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/introduction/overview.md)

MindSpore Transformers 自 **r2.0.0** 起以 **动态图（PyNative）实现** 作为演进主线。本章介绍动态图训练栈的整体架构、核心模块与训练能力，并给出最小落地入口。

> **动态图的能力边界**
>
> - 动态图实现源码位于 `mindformers/pynative/`。
> - 当前动态图聚焦 **预训练/微调** 训练场景；尚未覆盖的能力由静态图承载，清单见 [静态图实现特性](../feature/static_graph_features.md)。

---

## 概述

动态图实现采用 **分层、模块化** 的设计：

- **入口层**：统一脚本 `run_mindformer.py`，通过 `--mode 1` 路由到动态图训练器（源码 `run_mindformer.py` 中 `config.mode == 1` 分支）。
- **总控层**：`mindformers.pynative.trainer.Trainer` 负责构建模型/数据集/优化器，并驱动训练循环。
- **配置层**：以数据类（dataclass）形式集中管理 YAML 配置，加载时完成解析与校验。
- **能力层**：基于 MindSpore 的动态图能力实现多维并行、融合算子与显存优化。

下面先看一张「执行流程图」，再看一张「模块分层图」。

### 执行流程

```text
run_mindformer.py --mode 1            # 入口，PYNATIVE_MODE 路由
        │
        ▼
mindformers.pynative.trainer.Trainer  # 构建模型/数据集/优化器，驱动训练循环
        │
        ├── config/        YAML → dataclass 配置体系
        ├── base_models/   GPTModel（Dense/MoE 统一接口）
        ├── distributed/   多维并行与显存优化
        ├── optimizer/     AdamW/Muon
        ├── loss/          融合交叉熵
        ├── callback/      权重保存、Loss/指标监控
        └── tools/         监控与 Profiling
```

### 模块分层

模型侧自上而下分为「模型 → Transformer 组件 → 基础层」三层，与下方核心模块表一一对应：

```text
base_models/gpt/  GPTModel（Dense/MoE 统一接口，ModuleSpec 组装）
        │
        ▼
transformers/     Attention · MLA · MTP · TransformerLayer/Block · MLP · MoE
        │                                   (router/experts/shared_experts)
        ▼
layers/           Linear · RMSNorm · SwiGlu · FlashAttention · 掩码生成
```

> `base_models/common/embeddings` 提供 RoPE、YaRN 等位置编码，供上述各层调用。

---

## 核心模块

动态图实现各子模块及其职责如下：

| 模块 | 路径 | 职责 |
|------|------|------|
| 训练器 | `pynative/trainer/` | 训练总控 `Trainer`：构建模型/数据/优化器，执行前反向、梯度同步、保存与状态跟踪。 |
| 配置 | `pynative/config/` | 以 dataclass 集中管理配置，支持从 YAML 加载与校验。 |
| 分布式 | `pynative/distributed/` | 多维并行的设备网格构建与切分，以及多种显存优化。 |
| 优化器 | `pynative/optimizer/` | `AdamW` 与 `Muon` 优化器实现，支持分布式同步与混合精度。 |
| 损失 | `pynative/loss/` | 动态图融合交叉熵 `CrossEntropyLoss`，由自定义 `_LogSoftmax` + `_NLLLoss`（含手写反向）实现。 |
| 基础模型 | `pynative/base_models/gpt/` | 通用 `GPTModel`，统一支持 Dense 与 MoE，配合 `ModuleSpec` 机制按配置搭建模型；`common/embeddings` 提供 RoPE、YaRN 等位置编码。 |
| Transformer 组件 | `pynative/transformers/` | Attention、MLA、MTP、TransformerLayer/Block、MLP 及 MoE 子模块。 |
| 基础层 | `pynative/layers/` | Linear、LayerNorm/RMSNorm、SwiGlu、Flash Attention、掩码生成等融合算子。 |
| 回调 | `pynative/callback/` | Checkpoint 保存、Loss 监控、训练指标监控、MaxLogits 健康监测。 |
| 工具 | `pynative/tools/` | 指标监控聚合（`MonitorGroup`）与 Profiling。 |

下面对单元格信息较多的「配置」与「分布式」两个模块展开说明。

### 配置模块的 dataclass

`pynative/config/` 将 YAML 各段映射为独立的数据类，便于校验与默认值管理。常用配置类：

| dataclass | 对应职责 |
|---|---|
| `CheckpointConfig` | 权重保存/加载 |
| `TrainingConfig` | 训练步数、批大小、梯度累积等训练参数 |
| `ParallelismConfig` | 多维并行维度 |
| `OptimizerConfig` | 优化器类型与超参 |
| `LrSchedulerConfig` | 学习率策略 |
| `ModelConfig` | 模型结构参数 |
| `MonitorConfig` | 指标监控与可视化 |

### 分布式模块的能力

`pynative/distributed/` 同时承担「并行切分」与「显存优化」两类职责：

- **并行维度**：DP（含 FSDP/HSDP 参数切分）、TP、PP、CP、EP、SP。设备网格依据各维度乘积构建，满足 `dp_replicate * dp_shard * cp * tp * pp == world_size`（`parallel_dims.py`）。
- **显存优化**：重计算（activation checkpoint）、细粒度 SWAP、CPU offload。

> **关于 pet（LoRA）与 models 子目录**
>
> `pynative/pet/` 与 `pynative/models/` 目录当前仅含 `__init__.py`，尚无实现。LoRA 微调在动态图下**暂未实现**：触发时会在 `trainer/utils.py` 抛出 `NotImplementedError("Lora model is not implemented yet.")`。如需 LoRA，请使用静态图实现。

---

## 模型架构

动态图采用 **分层抽象 + 模块化** 的设计：以 `GPTModel`（General PreTrained Model）为统一模型接口，向下组合 `TransformerBlock`、`MoELayer`、`Attention`、`Linear`、`Embedding`、`Norm` 等模块化接口，并通过 `ModuleSpec` 机制自由组合搭建模型。所有模块基于 MindSpore 动态图进行了并行与算子融合优化。

动态图已覆盖 Dense 与 MoE（含 MLA、MTP）两类模型结构，已实现的模型清单见 [模型支持库](./models.md)。

---

## 训练能力

动态图训练栈提供以下能力（各能力一览见 [功能特性概述](../feature/overview.md)，配置说明页将随后续提交上线）：

- **多维混合并行**：数据并行（含 FSDP/HSDP 参数切分）、张量并行（TP）、流水线并行（PP，支持 1F1B 与 interleave）、上下文并行（CP，Colossal 方法）、专家并行（EP）与序列并行（SP）的灵活组合。
- **优化器与学习率**：AdamW、Muon；多种带 warmup 的学习率策略。
- **数据集**：Megatron 多源混合数据集（`BlendedMegatronDatasetDataLoader`，预处理后的 `.bin`/`.idx`）。
- **显存优化**：重计算（全量/选择性）、细粒度 SWAP、CPU offload。
- **权重**：Safetensors 格式的分片保存与加载，支持异步保存与冗余消除。
- **稳定性与可观测**：断点续训、梯度/参数范数与 Loss 监控、MaxLogits 数值健康监测与 Profiling。

---

## 下一步

读完架构后，最小落地路径如下（以下命令均在 **mindformers 仓库根目录**下执行）。

**单卡直跑**（调试/验证用）：

```bash
python run_mindformer.py --config <your_config.yaml> --mode 1
```

**多卡 msrun 拉起**（实际训练，以 8 卡为例）：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config <your_config.yaml> --mode 1"
```

`--mode 1` 即路由到动态图训练器。完整的「准备配置 → 启动 → 看结果」三步流程见 [快速开始](../quick_start/quick_start.md)，训练指南与各功能特性页正文将随后续提交上线。

---

## 相关文档

- 快速完成一个动态图训练任务：[快速开始](../quick_start/quick_start.md)
- 各能力一览：[功能特性概述](../feature/overview.md)
- 已支持的模型：[模型支持库](./models.md)
- 静态图提供的能力（推理/量化等）：[静态图实现](../static_graph/introduction/overview.md)

> 安装指南、训练指南与各功能特性页正文将随后续提交上线。
