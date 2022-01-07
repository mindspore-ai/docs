# CPU上推理

`CPU` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/multi_platform_inference_cpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 使用ONNX格式文件推理

与在GPU上进行推理类似，需要以下几个步骤：

1. 在训练平台上生成ONNX格式模型，具体步骤请参考[导出ONNX格式文件](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#onnx)。

2. 在CPU上进行推理，具体可以参考推理使用runtime/SDK的文档。如使用ONNX Runtime，可以参考[ONNX Runtime说明文档](https://github.com/microsoft/onnxruntime)。
