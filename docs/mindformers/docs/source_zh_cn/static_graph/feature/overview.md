<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} 已废弃（Deprecated）
:class: warning

本页属于 **静态图（GRAPH_MODE）实现** 章节，已标记为废弃。新特性优先在「r2.0.0 动态图实现」章节演进，请优先查阅动态图相关文档。
```

# 功能特性概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/static_graph/feature/overview.md)

MindSpore Transformers 在预训练、微调、推理与部署全流程中提供丰富的功能特性，便于用户进行配置化开发与调优。本章节按 **通用功能**、**训练功能** 和 **推理功能** 对全部功能进行分类汇总，便于快速查找与跳转。

## 通用功能

适用于预训练、微调与推理全流程的基础能力，便于统一配置与复用。

| 功能                                                                                                                  | 说明                                                  | 架构支持         |
|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|--------------|
| [启动任务](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/start_tasks.html)                             | 单卡、单机和多机任务一键启动。                                     | Mcore/Legacy |
| [Ckpt权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/ckpt.html)                                  | [Checkpoint 1.0 版本] 支持 ckpt 格式的权重文件转换及切分功能。         | Legacy       |
| [Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/safetensors.html)                    | [Checkpoint 1.0 版本] 支持 safetensors 格式的权重文件保存及加载功能。  | Mcore/Legacy |
| [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/configuration.html)                         | 支持使用 YAML 文件集中管理和调整任务中的可配置项。                        | Mcore/Legacy |
| [加载 Hugging Face 模型配置](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/load_huggingface_config.html) | 支持加载 Hugging Face 社区模型配置，即插即用、无缝对接。                 | Mcore        |
| [日志](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/logging.html)                                   | 日志相关介绍，包括日志结构、日志保存等。                                | Mcore/Legacy |
| [使用 Tokenizer](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/tokenizer.html)                       | Tokenizer 相关介绍，支持在推理、数据集中使用 Hugging Face Tokenizer。 | Mcore        |

## 训练功能

支持大规模、高可用的大模型训练与调优。

| 功能                                                                                                                         | 说明                                                  | 架构支持         |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|--------------|
| [数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/dataset.html)                                         | 支持多种类型和格式的数据集（Megatron、Hugging Face、MindRecord 等）。  | Mcore/Legacy |
| [训练超参数](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/training_hyperparameters.html)                      | 灵活配置大模型训练的超参数（学习率、优化器等）。                            | Mcore/Legacy |
| [训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/monitor.html)                                      | 提供大模型训练阶段的可视化服务，用于监控和分析训练过程中的各种指标和信息。               | Mcore/Legacy |
| [断点续训](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/resume_training.html)                                | [Checkpoint 1.0 版本] 支持 step 级断点续训，减少大规模训练意外中断造成的浪费。 | Mcore/Legacy |
| [checkpoint 保存和加载](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/checkpoint_saving_and_loading.html)      | [Checkpoint 2.0 版本] 支持 checkpoint 保存和加载功能。          | Mcore        |
| [断点续训 2.0](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/resume_training2.0.html)                         | [Checkpoint 2.0 版本] 支持 step 级断点续训及扩缩容、增量等场景。        | Mcore        |
| [训练高可用（Beta）](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/high_availability.html)                       | 提供临终 CKPT 保存、UCE 故障容错恢复和进程级重调度恢复等能力。                | Mcore        |
| [分布式并行训练](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/parallel_training.html)                           | 一键配置多维混合分布式并行，支持在万卡级集群中高效训练。                        | Mcore/Legacy |
| [训练内存优化](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/memory_optimization.html)                          | 支持重计算与细粒度激活值 SWAP，降低训练峰值显存。                         | Mcore/Legacy |
| [数据跳过和健康监测](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/skip_data_and_ckpt_health_monitor.html)         | 支持数据跳过与权重健康监测，提升训练鲁棒性。                              | Mcore/Legacy |
| [Pre-trained Model Average 权重合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/pma_fused_checkpoint.html) | 支持多 checkpoint 权重合并（PMA）及融合保存。                      | Mcore        |
| [其它训练特性](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/other_training_features.html)                      | 梯度累积、梯度裁剪、CPU 绑核、MoE DropRate、RoPE/SwiGLU 融合等。      | Mcore/Legacy |

## 推理功能

面向模型推理与部署场景，支持将训练好的模型高效部署并服务于生产环境。

| 功能                                                                                     | 说明                                              | 架构支持         |
|----------------------------------------------------------------------------------------|-------------------------------------------------|--------------|
| [量化](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/feature/quantization.html) | 集成 MindSpore Golden Stick 工具组件，提供统一量化推理流程，开箱即用。 | Mcore/Legacy |
