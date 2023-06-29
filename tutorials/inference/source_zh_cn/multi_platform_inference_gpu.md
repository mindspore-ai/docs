# GPU上推理

`Linux` `GPU` `推理应用` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/inference/source_zh_cn/multi_platform_inference_gpu.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 使用checkpoint格式文件推理

与在Ascend 910 AI处理器上推理一样。

## 使用ONNX格式文件推理

1. 在训练平台上生成ONNX格式模型，具体步骤请参考[导出ONNX格式文件](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/use/save_model.html#onnx)。

2. 在GPU上进行推理，具体可以参考推理使用runtime/SDK的文档。如在Nvidia GPU上进行推理，使用常用的TensorRT，可参考[TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt)。
