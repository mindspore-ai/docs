# 训练指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/guide/training.md)

## 概述

大模型预训练（Pretrain）是构建高性能语言模型的核心阶段，其本质是通过海量无标注数据使模型自主学习通用语言规律与知识。业界开源了许多各项指标优异的预训练模型，例如Llama、Qwen、DeepSeek系列模型，这些模型都在海量文本数据上学习"语言的概率分布"，使模型掌握词汇、语法、语义等通用能力，为下游任务（如问答、写作）提供扎实基础。

本质上，预训练通过反向传播算法优化模型参数，使损失函数最小化，从而提升模型对输入数据的预测或生成能力。MindSpore Transformers 提供了统一的预训练训练流程，结合生态提供了易用的解决方案。在统一训练流程中，启动训练任务可总结出如下关键步骤：

![/overview](../introduction/images/overall_architecture.png)

1. **任务前准备**：确定待训练模型配置、训练数据集准备，明确数据和模型两大关键点；
2. **修改训练配置**：基于已有硬件资源、模型以及数据，根据需求配置对应配置项。配置项涵盖基本配置、进阶配置以及高级配置，不同等级的配置项，允许训练任务完成不同的目标；
3. **启动训练任务**：基于已有硬件资源以及训练配置，通过快捷指令完成在不同集群规模下启动训练任务；
4. **训练状态监控**：在任务执行后，MindSpore Transformers提供各种手段观察训练状态，以供后续调试调优。

以下是MindSpore Transformers在LLM预训练任务上关键流程的具体描述。

> **动态图（PyNative）实现主线**
>
> 自 **r2.0.0** 起，MindSpore Transformers 以 **动态图（PyNative）实现** 作为演进主线，本文档默认面向动态图。动态图聚焦 **预训练** 训练场景；推理、服务化部署、量化等动态图尚未覆盖的能力，请查阅[静态图实现特性](../feature/static_graph_features.md)章节。

## 训练流程

### 1. 任务前准备

#### 确定指定规格模型

MindSpore Transformers支持了不同系列的预训练模型，例如DeepSeek系列以及Qwen3系列的一些典型规格。其中，动态图（PyNative）实现当前已支持 **DeepSeek-V3**（MoE + MLA + MTP）与 **Qwen3**（Dense）两类模型结构，对应实现位于 `mindformers/models/*/modeling_*_pynative.py`；其余既有模型为静态图实现，详见[模型支持库](../introduction/models.md)。

动态图采用 **分层抽象 + 模块化** 的设计：以 `GPTModel`（General PreTrained Model）为统一模型接口，向下组合 `TransformerBlock`、`MoELayer`、`Attention`、`Linear`、`Embedding`、`Norm` 等模块化接口，并通过 `ModuleSpec` 机制自由组合搭建模型。模型结构与超参通过 YAML 配置文件中 `model` 段显式配置（`model_type`、`architectures` 为固定字段，其余结构超参按透传交给模型类），动态图当前不支持通过 `pretrained_model_dir` 自动合并 HuggingFace `config.json`，结构参数须在 YAML 中显式写明。动态图整体架构详见[整体架构](../introduction/overview.md)。

#### 数据集预处理

在自然语言处理任务中，数据预处理是模型训练的关键前置环节。它不仅解决原始数据中的噪声、格式不一致等问题（如特殊字符、乱码等），更能通过结构化转换（如分词、向量化）将原始文本转化为模型可理解的数值形式。虽然广义的数据预处理可能包含收集、清洗、分词等全流程操作，但本环节默认输入数据已具备基础质量（即"干净"数据），因此重点聚焦于**分词转换**这一核心目标。

动态图模式下，数据集配置位于 YAML 配置文件的 `train_dataset` 字段中，目前支持以下两种数据集加载方式，覆盖常用开源与自定义场景：

- **Megatron 数据集**：支持加载符合 Megatron-LM 格式的数据集，适用于大规模语言模型的**预训练**任务。
- **MindRecord 数据集**：MindRecord 是 MindSpore 提供的高效数据存储/读取模块，支持将不同公开数据集转换为 MindRecord 格式进行训练。

具体处理方式与配置详见[数据集使用](../feature/dataset.md)。

**预训练数据处理**

针对 Megatron 数据集，MindSpore Transformers 提供了数据预处理脚本 [preprocess_indexed_dataset.py](https://atomgit.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py)，用于将 `json` 格式的原始文本语料转换成 `.bin` 或 `.idx` 文件。该方案支持多源混合：

- **灵活配置**：支持同时加载多个bin数据文件，并通过采样比例参数控制不同数据源的混合权重；
- **高效训练**：二进制存储格式大幅提升了IO效率，特别适合大规模预训练场景。

预处理完成后，可通过配置 `BlendedMegatronDatasetDataLoader` 加载 Megatron 格式数据集进行预训练，详见[数据集使用-Megatron数据集章节](../feature/dataset.md#megatron-数据集)。此外，MindRecord 格式数据集也支持通过 `MultiSourceDataLoader` 实现多源数据集高效加载与采样，详见[数据集使用-MindRecord数据集章节](../feature/dataset.md#mindrecord-数据集)。

### 2. 配置文件准备

在进行一次预训练任务时，大模型参数量庞大（通常数十亿至万亿级），需依赖分布式计算资源高效训练以及对各项超参的修改，用于保证任务的正常执行及模型的最终性能指标。动态图（PyNative）训练使用一个 YAML 文件集中管理所有可配置项，由 [mindformers/pynative/config/config.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/pynative/config/config.py) 的 **dataclass 配置体系**（`TrainConfig` 及其子配置类）对 YAML 加载时进行解析与校验。一份完整配置主要由以下顶层段组成（详见[配置文件说明](../feature/configuration.md)）：

- **模型配置**（`model`）：根据预定的模型规格，修改配置文件中与模型架构相关的参数，如层数、头数、隐藏层维度等；
- **数据配置**（`train_dataset`）：指定预处理得到的数据集，配置数据集路径、数据加载方式等；
- **训练超参**（`training`、`optimizer`、`lr_scheduler`）：根据模型训练策略，指定优化器类型、损失函数、学习率、数据批量大小、训练轮数等；
- **并行策略**（`parallelism`）：根据集群规模及模型参数，配置运用数据并行（含FSDP/HSDP）、张量并行（TP）、流水线并行（PP）、上下文并行（CP）、专家并行（EP）等技术使超大规模模型能够正常训练或进行性能调优；
- **状态监控**（`monitor`、`profiler`）：配置loss打印步数间隔、配置profiling采集性能数据；精度调试任务中配置打印/可视化关键数值用来定位精度问题，例如local norm、local loss、优化器状态等；
- **高可用相关**（`checkpoint`）：配置权重保存步数、断点续训权重、均衡加载等高可用特性，保障在训练过程中能够平稳运行。

根据配置类型与预训练场景的不同，MindSpore Transformers 将可配置参数进行分层，并说明各层配置的适用场景与预期目标。详细信息如下：

<table>
  <tr>
    <th>配置类型</th>
    <th>类型说明</th>
    <th>配置项</th>
    <th>配置指导</th>
  </tr>
  <tr>
    <td rowspan="3">基础配置</td>
    <td rowspan="3">通过配置该部分配置，能够基于当前模型结构下，拉起一个简单的训练任务</td>
    <td>数据集</td>
    <td><a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html target="_blank">数据集使用</a></td>
  </tr>
  <tr>
    <td>并行配置</td>
    <td>
    <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html#parallelism-——-多维并行 target="_blank">并行配置项说明</a><br>
    <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/parallel_training.html target="_blank">分布式并行训练指南</a>
    </td>
  </tr>
  <tr>
    <td>训练超参</td>
    <td>
    <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/training_hyperparameters.html target="_blank">训练超参数与优化器</a><br>
    <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html target="_blank">其它训练特性（梯度累积/裁剪/融合算子/混合精度）</a>
    </td>
  </tr>
  <tr>
    <td rowspan="3">高级配置</td>
    <td rowspan="3">通过配置该部分，可感知训练状态并保障多次训练任务的连贯执行</td>
    <td>权重保存</td>
    <td>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/save_load_checkpoint.html target="_blank">权重保存与加载（Safetensors）</a><br>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html#checkpoint-——-权重保存与加载 target="_blank">Callbacks配置CheckpointMonitor</a>
    </td>
  </tr>
  <tr>
    <td>断点续训</td>
    <td>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/resume_training.html target="_blank">断点续训示例</a>
    </td>
  </tr>
  <tr>
    <td>在线监控</td>
    <td>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/monitor.html target="_blank">训练指标监控与 Profiling</a>
    </td>
  </tr>
  <tr>
    <td rowspan="2">进阶配置</td>
    <td rowspan="2">通过配置该部分，可进行训练过程的状态监测与性能调优，实现在不同集群规模下的稳定高性能训练</td>
    <td>性能调优</td>
    <td>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/memory_optimization.html target="_blank">训练内存优化</a>
    </td>
  </tr>
  <tr>
    <td>其他训练特性</td>
    <td>
      <a href=https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/other_training_features.html#%E4%BA%8C%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA target="_blank">梯度累积/裁剪/融合算子/混合精度</a>
    </td>
  </tr>
</table>

除以上配置项外，训练任务的所有配置项由[配置文件](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html)统一控制，可根据配置项说明灵活调整设置。

> **动态图与静态图配置差异提示**
>
> - 动态图配置项主要集中在上述 14 个顶层段，与静态图 YAML 字段不完全一致，请勿直接复用静态图配置；
> - 动态图**仅用 Safetensors 格式**保存与加载权重，不涉及 ckpt 与 Safetensors 之间的格式转换；
> - `model` 段中的超参须显式定义，不支持自动合并 HuggingFace 社区中的 `config.json` 文件。

### 3. 启动训练任务

MindSpore Transformers动态图训练支持单机多卡、多机多卡分布式训练，集群规模支持从单机8卡至万卡的超大规模分布式训练。动态图训练统一通过 [run_mindformer.py](https://atomgit.com/mindspore/mindformers/blob/master/run_mindformer.py) 拉起，须通过 `--mode 1` 显式指定使用 PYNATIVE_MODE。

- **单卡启动**：使用 `run_mindformer.py` 拉起任务，可通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 选择具体某张卡：

  `ASCEND_RT_VISIBLE_DEVICES` 用于控制当前进程可见的 NPU 物理卡，取值为卡号或卡号列表。例如 `=0` 表示只让进程看到 0 号物理卡；若想使用 3 号卡则写 `=3`；多卡时可写成 `=0,1,2,3` 这样的列表形式。若不显式指定，默认使用所有卡。

  ```bash
  # 指定使用 7 号卡执行单卡训练
  export ASCEND_RT_VISIBLE_DEVICES=7
  python run_mindformer.py --config /path/to/your.yaml --mode 1
  ```

- **单机多卡/多机多卡启动**：通过 `scripts/msrun_launcher.sh` 脚本基于 [msrun](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html) 工具拉起分布式任务：

  ```bash
  # 单机8卡
  bash scripts/msrun_launcher.sh "run_mindformer.py --config /path/to/your.yaml --mode 1" 8
  ```

具体启动方式（含多机多卡、自定义脚本启动、启动期排障等）参照[启动任务](../feature/start_task.md)文档。最小可运行端到端示例（DeepSeek-V3 2卡预训练）参照[快速开始](../quick_start/quick_start.md)。

### 4. 训练状态监控

当预训练周期较长（数周至数月）时，需要实时监控关键指标并动态调整，以保障最终训练得到的模型能够达到预期的效果。此时，需要关注的状态项主要有：

- **性能指标**：每秒训练token数/样本数（吞吐量）、NPU利用率（算力利用率）、每step耗时；
- **精度指标**：损失函数值、梯度范数（防爆炸/消失）、MaxLogits 数值健康监测；
- **检查点检查**：定期保存模型中间状态（如每N步），防止训练中断导致数据丢失。

针对不同的监控值，MindSpore Transformers 在动态图训练过程中会打印详尽的日志用于查看中间过程状态，并提供以下监控手段：

- **训练指标监控**：通过 `monitor` 段配置 grad/param 范数、Loss 监控、MoE 监控等，对应配置详见[配置文件说明-monitor章节](../feature/configuration.md#monitor-——-训练监控)；
- **性能分析**：通过 `profiler` 段开启 profiling 数据采集（算子耗时、内存、调用栈等），对应配置详见[配置文件说明-profiler章节](../feature/configuration.md#profiler-——-性能分析)；

权重在中间保存检查点或训练完成后，模型权重将保存至 `save_path` 指定路径下，每次保存会在 `save_path` 下生成一个按 step 命名的子目录，内含 Safetensors 权重分片、续训元信息 `common.json` 与分片布局元数据 `metadata.json`。后续可以使用保存的权重进行续训等，详见[权重保存与加载](../feature/save_load_checkpoint.md)。

## 训练实践

MindSpore Transformers 提供了更为细致的预训练流程及实践：

- **快速开始**：以 DeepSeek-V3 为例，2 卡执行最小规模预训练的端到端流程，详见[快速开始](../quick_start/quick_start.md)；
- **断点续训实践**：step 级断点续训，含标准续训、仅权重初始化、改 batch size 续训、分布式均衡加载等 4 个场景，详见[断点续训](../feature/resume_training.md)。
