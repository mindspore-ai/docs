<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} 已废弃（Deprecated）
:class: warning

本页属于 **静态图（GRAPH_MODE）实现** 章节，已标记为废弃。新特性优先在「r2.0.0 动态图实现」章节演进，请优先查阅动态图相关文档。
```

# 高阶开发概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/static_graph/advanced_development/overview.md)

MindSpore Transformers 高阶开发面向模型迁移、调优与精度验证等进阶场景，帮助用户在完成基础训练与推理后，进一步做开发迁移、调试调优和精度对比。本章节按 **调试调优**、**模型开发与配置**、**精度对比** 和 **API 参考** 对全部高阶开发文档进行分类汇总，便于快速查找与跳转。

## 调试调优

面向训练与推理过程中的精度与性能问题，提供系统化的排查与优化方法。

| 文档                                                                                                                | 说明                                                     | 架构支持         |
|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|--------------|
| [精度调优](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/precision_optimization.html)   | 大模型训练常见精度问题与通用定位方法，包括 CheckList、参数对齐、随机性固定及长稳排查等。      | Mcore/Legacy |
| [性能调优](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/performance_optimization.html) | 大模型性能调优思路与工具，涵盖数据加载、前向/反向计算、通信与调度等环节的优化及 Profile 使用指导。 | Mcore/Legacy |

## 模型开发与配置

支持从零构建或迁移模型，以及通过配置模板快速拉起训练与推理任务。

| 文档                                                                                                                           | 说明                                                                             | 架构支持   |
|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------|
| [开发迁移](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/dev_migration.html)                       | 基于 MindSpore Transformers 构建大模型的完整流程，包括配置、模型、分词器与 YAML 配置的编写与适配。               | Legacy |
| [推理配置模板使用指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/yaml_config_inference.html)         | 推理场景下 YAML 配置模板的使用方法，支持按 Hugging Face/ModelScope 模型目录快速配置并启动推理。                | Mcore  |
| [训练配置模板使用说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/training_template_instruction.html) | 预训练与微调的通用配置模板说明，支持稠密/MoE 等场景，便于未支持规格或自定义模型的训练任务快速启动。                           | Mcore  |
| [权重转换开发适配](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/weight_transfer.html)                 | 将新模型适配 MindSpore Transformers 权重转换功能，实现 Hugging Face 权重到 MindSpore 权重的统一转换与加载。 | Mcore  |

## 精度对比

用于验证与标杆实现或 GPU 环境在训练与推理层面的一致性。

| 文档                                                                                                                           | 说明                                                     | 架构支持  |
|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|-------|
| [与 Megatron-LM 比对训练精度](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/accuracy_comparison.html) | 与 Megatron-LM 在模型层面的训练精度对齐方法，包括等价结构构建、前向/损失/梯度等关键指标对比。 | Mcore |
| [推理精度比对](https://www.mindspore.cn/mindformers/docs/zh-CN/master/static_graph/advanced_development/inference_precision_comparison.html)    | 推理精度验收流程与定位思路，涵盖在线推理验证、数据集评测及常见问题的排查与解决。               | Mcore |

## API 参考

MindSpore Transformers 各模块的 API 文档入口。

| 文档                                                                                          | 说明                                |
|---------------------------------------------------------------------------------------------|-----------------------------------|
| [API](https://www.mindspore.cn/mindformers/docs/zh-CN/master/api.html) | mindformers 及各子模块的 API 索引与详细接口说明。 |
