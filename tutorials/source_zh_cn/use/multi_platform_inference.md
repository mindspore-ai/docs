# 多平台推理

<a href="https://gitee.com/mindspore/docs/blob/r0.3/tutorials/source_zh_cn/advanced_use/multi_platform_inference.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

基于MindSpore训练后的模型，支持在不同的硬件平台上执行推理。本文介绍各平台上的推理流程。

## Ascend 910 AI处理器上推理

MindSpore提供了`model.eval()`接口来进行模型验证，你只需传入验证数据集即可，验证数据集的处理方式与训练数据集相同。完整代码请参考<https://gitee.com/mindspore/mindspore/blob/r0.3/example/resnet50_cifar10/eval.py>。

```python
res = model.eval(dataset)
```

此外，也可以通过`model.predict()`接口来进行推理操作，详细用法可参考API说明。

## Ascend 310 AI处理器上推理

1. 参考[模型导出](https://www.mindspore.cn/tutorial/zh-CN/0.3.0-alpha/use/saving_and_loading_model_parameters.html#geironnx)生成ONNX或GEIR模型。

2. 云上环境请参考[Ascend910训练和Ascend310推理的样例](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0026.html)完成推理操作。裸机环境（对比云上环境，即本地有Ascend 310 AI 处理器）请参考Ascend 310 AI处理器配套软件包的说明文档。

## GPU上推理

1. 参考[模型导出](https://www.mindspore.cn/tutorial/zh-CN/0.3.0-alpha/use/saving_and_loading_model_parameters.html#geironnx)生成ONNX模型。

2. 参考[TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt)，在Nvidia GPU上完成推理操作。

## 端侧推理

端侧推理需使用MindSpore Predict推理引擎，详细操作请参考[端侧推理教程](https://www.mindspore.cn/tutorial/zh-CN/0.3.0-alpha/advanced_use/on_device_inference.html)。