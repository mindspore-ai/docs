# Inference on a CPU

`Linux` `CPU` `Inference Application` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Inference on a CPU](#inference-on-a-cpu)
    - [Inference Using a Checkpoint File](#inference-using-a-checkpoint-file)
    - [Inference Using an ONNX File](#inference-using-an-onnx-file)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/multi_platform_inference_cpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Inference Using a Checkpoint File

The inference is the same as that on the Ascend 910 AI processor.

## Inference Using an ONNX File

Similar to the inference on a GPU, the following steps are required:

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/docs/programming_guide/en/r1.3/save_model.html#export-onnx-model).

2. Perform inference on a CPU by referring to the runtime or SDK document. For details about how to use the ONNX Runtime, see the [ONNX Runtime document](https://github.com/microsoft/onnxruntime).
