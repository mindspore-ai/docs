# Quantization

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/quantization.md)

## Overview

Quantization is an important technology for compressing foundation models. It converts floating-point parameters in a model into low-precision integer parameters to compress the parameters. As the parameters and specifications of a model increase, quantization can effectively reduce the model storage space and loading time during model deployment, improving the model inference performance.

MindSpore Transformers integrates the MindSpore Golden Stick tool component to provide a unified quantization inference process, facilitating out-of-the-box use. Please refer to [MindSpore Golden Stick Installation Tutorial](https://www.mindspore.cn/golden_stick/docs/en/r1.0.0/install.html) for installation and [MindSpore Golden Stick Application PTQ algorithm](https://www.mindspore.cn/golden_stick/docs/en/r1.0.0/ptq/ptq.html) to quantify the models in MindSpore Transformers.

## Model Support

Currently, only the following models are supported, and the supported models are continuously being added.

| Supported Model                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------|
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek3/deepseek3_671b/predict_deepseek3_671b.yaml)     |
| [DeepSeek-R1](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml) |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/predict_llama2_13b_ptq.yaml)                             |