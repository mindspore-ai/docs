# 静态图实现特性

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/static_graph_features.md)

> **日落特性（Sunset）**
>
> 以下能力当前**仅在静态图（GRAPH_MODE）实现**，动态图（PyNative）尚未支持，标记为日落特性。新功能优先在动态图演进，下列特性请前往「静态图实现」章节查阅。

动态图（r2.0.0）当前聚焦**训练**全流程，且模型结构通过 YAML `model` 段显式配置。以下能力暂未在动态图实现，需使用静态图：

| 特性 | 说明 | 跳转 |
|------|------|------|
| 推理 | 训练后模型的推理流程。 | [静态图 · 推理](../static_graph/guide/inference.md) |
| 服务化部署 | 基于 vLLM 等的服务化部署。 | [静态图 · 服务化部署](../static_graph/guide/deployment.md) |
| 量化 | 集成 MindSpore Golden Stick 的量化推理。 | [静态图 · 量化](../static_graph/feature/quantization.md) |
| Ckpt 权重 | ckpt 格式权重转换/切分（Checkpoint 1.0）。 | [静态图 · Ckpt 权重](../static_graph/feature/ckpt.md) |
| 加载 HF 模型配置 | 通过 `pretrained_model_dir` 自动合并 HF `config.json`（动态图需在 `model` 段显式写结构）。 | [静态图 · 加载 HF 模型配置](../static_graph/feature/load_huggingface_config.md) |

> 上述能力在动态图实现后，将从本页迁出对应条目并补充动态图文档。
