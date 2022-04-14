# 骨骼检测模型

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/docs/source_zh_cn/posenet_lite.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 骨骼检测介绍

骨骼检测可以识别摄像头中，不同姿势下人体的面部五官与肢体姿势。

使用骨骼检测模型的输出如图：

蓝色标识点检测人体面部的五官分布及上肢、下肢的骨骼走势。此次推理置信分数0.98/1，推理时延66.77ms。

![image_posenet](images/posenet_detection.png)

使用MindSpore Lite实现骨骼检测的[示例代码](https://gitee.com/mindspore/models/tree/r1.7/official/lite/posenet)。
