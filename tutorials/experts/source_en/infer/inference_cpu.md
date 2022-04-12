# Inference on a CPU

`CPU` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/model_infer/multi_platform_inference_cpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Inference Using an ONNX File

Similar to the inference on a GPU, the following steps are required:

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorials/experts/en/master/model_infer/save_model.html#export-onnx-model).

2. Perform inference on a CPU by referring to the runtime or SDK document. For details about how to use the ONNX Runtime, see the [ONNX Runtime document](https://github.com/microsoft/onnxruntime).
