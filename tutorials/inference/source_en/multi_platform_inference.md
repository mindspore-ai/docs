# Multi-Platform Inference Overview

`Linux` `Ascend` `GPU` `CPU` `Inference Application` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_en/multi_platform_inference.md)

Models trained by MindSpore support the inference on different hardware platforms. This document describes the inference process on each platform.

The inference can be performed in either of the following methods based on different principles:

- Use a checkpoint file for inference. That is, use the inference API to load data and the checkpoint file for inference in the MindSpore training environment.
- Convert the checkpoint file into a common model format, such as ONNX or AIR, for inference. The inference environment does not depend on MindSpore. In this way, inference can be performed across hardware platforms as long as the platform supports ONNX or AIR inference. For example, models trained on the Ascend 910 AI processor can be inferred on the GPU or CPU.

MindSpore supports the following inference scenarios based on the hardware platform:

| Hardware Platform       | Model File Format | Description                              |
| ----------------------- | ----------------- | ---------------------------------------- |
| Ascend 910 AI processor | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| Ascend 310 AI processor | ONNX or AIR       | Equipped with the ACL framework and supports the model in OM format. You need to use a tool to convert a model into the OM format. |
| GPU                     | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| GPU                     | ONNX              | Supports ONNX Runtime or SDK, for example, TensorRT. |
| CPU                     | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| CPU                     | ONNX              | Supports ONNX Runtime or SDK, for example, TensorRT. |

> - Open Neural Network Exchange (ONNX) is an open file format designed for machine learning. It is used to store trained models. It enables different AI frameworks (such as PyTorch and MXNet) to store model data in the same format and interact with each other. For details, visit the ONNX official website <https://onnx.ai/>.
> - Ascend Intermediate Representation (AIR) is an open file format defined by Huawei for machine learning and can better adapt to the Ascend AI processor. It is similar to ONNX.
> - Ascend Computer Language (ACL) provides C++ API libraries for users to develop deep neural network applications, including device management, context management, stream management, memory management, model loading and execution, operator loading and execution, and media data processing. It matches the Ascend AI processor and enables hardware running management and resource management.
> - Offline Model (OM) is supported by the Huawei Ascend AI processor. It implements preprocessing functions that can be completed without devices, such as operator scheduling optimization, weight data rearrangement and compression, and memory usage optimization.
> - NVIDIA TensorRT is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime to improve the inference speed of the deep learning model on edge devices. For details, see <https://developer.nvidia.com/tensorrt>.
