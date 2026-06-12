# 功能特性概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/overview.md)

MindSpore Transformers 动态图（PyNative）实现面向 **预训练/微调** 训练全流程提供丰富的功能特性，便于配置化开发与调优。本章按 **通用功能** 与 **训练功能** 分类汇总；推理与部署相关能力当前由静态图提供，集中列在「静态图实现特性」。

> **各特性页正文上线中**
>
> 下表所列各特性的详细文档将随后续提交上线，届时本页将补充对应跳转链接。

## 通用功能

| 功能 | 说明 |
|------|------|
| 启动任务 | 基于 `msrun` 的单卡、单机多卡与多机任务一键启动。 |
| 配置文件说明 | dataclass 风格 YAML，集中管理训练全部可配置项。 |
| 日志 | 日志结构与保存说明。 |

## 训练功能

| 功能 | 说明 |
|------|------|
| 数据集 | Megatron 数据集（`BlendedMegatronDatasetDataLoader`），预处理后的 `.bin`/`.idx`，支持多源混合。 |
| 训练超参数与优化器 | AdamW / Muon 优化器与带 warmup 的学习率策略。 |
| 分布式并行训练 | DP/FSDP、TP、PP、CP、EP、SP 多维混合并行。 |
| 训练内存优化 | 重计算（full/select）、细粒度 SWAP、CPU offload。 |
| Safetensors 权重 | Safetensors 分片保存与加载，支持异步保存与冗余消除。 |
| 断点续训 | step 级断点续训，减少大规模训练中断损失。 |
| 训练指标监控与 Profiling | grad/param 范数、Loss 监控、MaxLogits 数值健康监测与性能分析。 |
| 其它训练特性 | 梯度累积、梯度裁剪、融合算子等。 |

## 静态图实现特性

推理、部署等尚未在动态图覆盖的能力清单，见 [静态图实现特性](./static_graph_features.md)。
