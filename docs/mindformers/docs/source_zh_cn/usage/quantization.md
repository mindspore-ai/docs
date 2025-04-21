# 量化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/usage/quantization.md)

## 概述

量化（Quantization）作为一种重要的大模型压缩技术，通过对模型中的浮点参数转为低精度的整数参数，实现对参数的压缩。随着模型的参数和规格不断增大，量化在模型部署中能有效减少模型存储空间和加载时间，提高模型的推理性能。

MindSpore Transformers 集成 MindSpore Golden Stick 工具组件，提供统一量化推理流程，方便用户开箱即用。请参考 [MindSpore Golden Stick 安装教程](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/install.html)进行安装，并参考 [MindSpore Golden Stick 应用PTQ算法](https://www.mindspore.cn/golden_stick/docs/zh-CN/r1.1.0/ptq/ptq.html)对MindSpore Transformers中的模型进行量化。

## 模型支持度

当前仅支持以下模型，支持模型持续补充中。

| 支持的模型                                                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------------|
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml)     |
| [DeepSeek-R1](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml) |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/predict_llama2_13b_ptq.yaml)                             |