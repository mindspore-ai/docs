# 整体架构

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/introduction/overview.md)

## 概述

MindSpore Transformers 整体架构如下：

![/overall_architecture](./images/overall_architecture.png)

MindSpore Transformers 北向支持用户集成在自有训推平台或者开源组件中，支持昇腾自有技术栈外也积极拥抱开源社区，具体如下：

1. 训练平台：MindCluster、第三方平台
2. 服务化组件：vLLM
3. 社区：魔乐社区、Hugging Face

MindSpore Transformers 南向基于昇思+昇腾的大模型技术栈，利用昇思框架结合 CANN 对昇腾硬件进行亲和优化，提供高性能的模型训推体验。

对于 MindSpore Transformers 本身，主要分为如下模块：

1. 大模型训练、推理统一调度入口层：提供统一的运行工具脚本 msrun_launcher.sh，统一执行与调度套件内所有模型的分布式训推流程。
2. 注册/配置层：按接口类型实现类工厂，使能高阶接口层按配置初始化对应的任务接口、模型接口。
3. 大模型模型库：实现高性能大模型库以及基础 Transformer 接口，即可支持用户配置化构建自有模型，也可自定义开发，可满足不同开发场景。
4. 数据集：实现大模型训练、微调任务的数据加载封装，可原生支持 Hugging Face Datasets、Megatron 数据集以及 MindSpore 原生 MindRecord 的数据支持。
5. 训练组件：实现训练流程的基础接口，包含学习率策略、优化器、训练回调以及 TrainOneStepWrapper 等接口。
6. 工具层：独立工具脚本，目前提供数据预处理、Hugging Face 权重互转、评测工具脚本。
7. DFX（Design for X）：实现故障诊断、故障监测等高可用特性，降低训练故障恢复成本。

## 模型架构

MindSpore Transformers 在 1.6.0 版本之后应用了全新的模型架构，原有架构（标记为 Legacy）各模型单独有一份模型代码，较难维护与优化。新架构（标记为 Mcore）对通用 Transformer 架构大模型进行分层抽象与模块化实现，涉及下层的基础层，如 Linear、Embedding、Norm 等，以及上层的 MoELayer、TransformerBlock 和模型统一接口 GPTModel（General PreTrained Model）等。所有模块化接口基于 MindSpore 提供的并行能力，进行了深度并行优化，对外提供开箱即用的高性能接口。所有高度封装集成的接口支持通过 ModuleSpec 机制自由组合进行模型搭建。

## 训练能力

MindSpore Transformer 训练提供了一系列高效易用特性以及生态协同能力，协助用户在大模型的预训练和微调环节实现简洁易用、高效稳定。对外能力涵盖：

- 多维混合并行，包含数据并行、模型并行、优化器并行、流水线并行、序列并行、上下文并行、MoE 专家并行等；
- 预训练阶段支持直接加载 Megatron-LM 多源混合数据集，避免跨平台和框架的数据集迁移问题；
- 微调阶段接入 Hugging Face 生态能力，支持使用 Hugging Face SFT 数据集，支持使用 Hugging Face Tokenizer 实现数据预处理，支持读取 Hugging Face 模型配置实例化模型，支持加载原生 Hugging Face Safetensors 权重。配合零代码、配置化使能低参微调的能力，实现高效便捷微调；
- 支持分布式权重自动切分加载，在分布式策略切换调试、集群扩缩容等场景下，无需手动转换权重，助力高效调试与训练；
- 提供训练状态监控、故障快恢、异常跳过、断点续训等易用性和高可用特性，支持预训练/微调过程中的可测试性、可维护性和可靠性；
- 封装了高性能基础接口，接口设计与 Megatron-LM 对齐，计算精度对齐达标。结合模型迁移和精度比对相关的教程文档，以及昇腾工具链提供的 Cell 级 dump 工具，实现低门槛、高效率的模型迁移与构建。

## 推理能力

MindSpore Transformers 推理北向对接第三方开源组件，为开发者提供更丰富的推理部署、量化和评测能力：

- 支持直接加载使用 Hugging Face 开源配置、权重和 tokenizer，一键启动推理；
- 支持对接 vLLM 服务化框架，实现推理服务化部署。支持 Continuous Batch、Prefix Cache、Chunked Prefill 等服务化特性；
- 通过 MindSpore Golden-Stick 量化套件，Legacy模型可以实现A16W8、A8W8、A8W4量化推理，Mcore 模型预计下版本支持A8W8、A8W4量化推理；
- 通过 AISbench 评测套件，接入 vLLM 服务化的 MindSpore Transformers 模型，可以实现CEval、GSM8K、AIME 等 20+ 主流榜单评测。

南向依靠 MindSpore 框架提供的推理优化能力，实现高性能推理：

- 依靠框架 Runtime 运行时提供的多级流水下发特性，在 host 侧将算子调度拆分成 InferShape、Resize 和 Launch 三个任务流水下发，充分利用 host 多线程资源，提升算子下发效率，从而实现推理加速；
- 默认采用 PyNative 编程模式 + JIT 即时编译技术，将模型编译成静态计算图进行推理加速，也可以一键切换 PyNative 动态图模式便于开发调试；
- MindSpore Transformers 支持使用 ACLNN、ATB和 MindSpore 提供的推理加速/融合算子，在昇腾底座上实现更加高效的推理性能。
