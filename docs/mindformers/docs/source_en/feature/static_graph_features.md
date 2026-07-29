# Static Graph Features

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/static_graph_features.md)

> **Sunset Features**
>
> The following capabilities are currently **implemented only in static graphs (GRAPH_MODE)** and are not supported in dynamic graphs (PyNative). They are marked as sunset features. New features are preferentially evolved in dynamic graphs. For details about the following features, see section "Static Graph Implementation."

Dynamic graphs (r2.0.0) currently focus on the entire **training** process, and the model structure is explicitly configured through the YAML `model` section. The following capabilities are not yet implemented in dynamic graphs and require static graphs.

| Feature        | Description                                                                     | Link                                                                    |
|------------|-------------------------------------------------------------------------|------------------------------------------------------------------------|
| Inference        | Inference process of a trained model.                                                            | [Static Graph > Inference](../static_graph/guide/inference.md)                        |
| Service deployment     | Service deployment based on vLLM and others.                                                       | [Static Graph > Service Deployment](../static_graph/guide/deployment.md)                    |
| Quantification        | Quantization inference integrated with MindSpore Golden Stick.                                       | [Static Graph > Quantization](../static_graph/feature/quantization.md)                   |
| Checkpoint weight   | Converts or shards weights in CKPT format, for example, Checkpoint 1.0.                                        | [Static Graph > Ckpt Weights](../static_graph/feature/ckpt.md)                      |
| Loading the HF model configuration| Automatically merges HF `config.json` through `pretrained_model_dir` (the dynamic graph structure must be explicitly written in the `model` section). | [Static Graph > Loading Hugging Face Model Configuration](../static_graph/feature/load_huggingface_config.md)|

> After the preceding capabilities are implemented in dynamic graphs, the corresponding items will be moved from this page and supplemented in the dynamic graph documentation.
