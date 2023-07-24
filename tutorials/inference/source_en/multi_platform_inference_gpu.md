# Inference on a GPU

`Linux` `GPU` `Inference Application` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/inference/source_en/multi_platform_inference_gpu.md)

## Inference Using a Checkpoint File

The inference is the same as that on the Ascend 910 AI processor.

## Inference Using an ONNX File

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.2/use/save_model.html#export-onnx-model).

2. Perform inference on a GPU by referring to the runtime or SDK document. For example, use TensorRT to perform inference on the NVIDIA GPU. For details, see [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).
