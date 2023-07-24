# 骨骼检测模型

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_zh_cn/posenet_lite.md)

## 骨骼检测介绍

骨骼检测可以识别摄像头中，不同姿势下人体的面部五官与肢体姿势。

使用骨骼检测模型的输出如图：

蓝色标识点检测人体面部的五官分布及上肢、下肢的骨骼走势。此次推理置信分数0.98/1，推理时延66.77ms。

![image_posenet](images/posenet_detection.png)

使用MindSpore Lite实现骨骼检测的[示例代码](https://gitee.com/mindspore/mindspore/tree/r1.2/model_zoo/official/lite/posenet)。
