# Inference on a GPU

`Linux` `GPU` `Inference Application` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/multi_platform_inference_gpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Inference Using a Checkpoint File

The inference is the same as that on the Ascend 910 AI processor.

## Inference Using an ONNX File

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/docs/programming_guide/en/r1.3/save_model.html#export-onnx-model).

2. Perform inference on a GPU by referring to the runtime or SDK document. For example, use TensorRT to perform inference on the NVIDIA GPU. For details, see [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).
