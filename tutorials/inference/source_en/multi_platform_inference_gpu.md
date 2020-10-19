# Inference on a GPU

`Linux` `GPU` `Inference Application` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Inference on a GPU](#inference-on-a-gpu)
    - [Inference Using a Checkpoint File](#inference-using-a-checkpoint-file)
    - [Inference Using an ONNX File](#inference-using-an-onnx-file)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/inference/source_en/multi_platform_inference_gpu.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Inference Using a Checkpoint File

The inference is the same as that on the Ascend 910 AI processor.

## Inference Using an ONNX File

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/master/use/save_model.html#export-onnx-model).

2. Perform inference on a GPU by referring to the runtime or SDK document. For example, use TensorRT to perform inference on the NVIDIA GPU. For details, see [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).
