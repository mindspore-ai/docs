# Ascend 310 AI处理器上推理

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_zh_cn/multi_platform_inference_ascend_310.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 使用ONNX与AIR格式文件推理

Ascend 310 AI处理器上搭载了ACL框架，他支持OM格式，而OM格式需要从ONNX或者AIR模型进行转换。所以在Ascend 310 AI处理器上推理，需要下述两个步骤：

1. 在训练平台上生成ONNX或AIR格式模型，具体步骤请参考[导出AIR格式文件](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_model.html#air)和[导出ONNX格式文件](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_model.html#onnx)。

2. 将ONNX/AIR格式模型文件，转化为OM格式模型，并进行推理。
   - 云上（ModelArt环境），请参考[Ascend910训练和Ascend310推理的样例](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0026.html)完成推理操作。
   - 本地的裸机环境（对比云上环境，即本地有Ascend 310 AI 处理器），请参考Ascend 310 AI处理器配套软件包的说明文档。