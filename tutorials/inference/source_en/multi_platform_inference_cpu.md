# Inference on a CPU

`Linux` `CPU` `Inference Application` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_en/multi_platform_inference_cpu.md" target="_blank"><img src="./_static/logo_source.png"></a>


## Inference Using a Checkpoint File
The inference is the same as that on the Ascend 910 AI processor.

## Inference Using an ONNX File
Similar to the inference on a GPU, the following steps are required:

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_model.html#export-onnx-model).

2. Perform inference on a CPU by referring to the runtime or SDK document. For details about how to use the ONNX Runtime, see the [ONNX Runtime document](https://github.com/microsoft/onnxruntime).
