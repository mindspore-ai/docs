# 对象检测

<a href="https://gitee.com/mindspore/docs/blob/master/lite/docs/source_zh_cn/object_detection.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 对象检测介绍

对象检测可以识别出图片中的对象和该对象在图片中的位置。 如：对下图使用对象检测模型的输出如下表所示，使用矩形框识别图中对象的位置并且标注出对象类别的概率，其中坐标中的4个数字分别为Xmin，Ymin,，Xmax,，Ymax；概率表示反应被检测物理的可信程度。

![image_classification](images/object_detection.png)

| 类别  | 概率 | 坐标             |
| ----- | ---- | ---------------- |
| mouse | 0.78 | [10, 25, 35, 43] |

使用MindSpore Lite实现对象检测的[示例代码](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/lite/object_detection)。

## 对象检测模型列表

下表是使用MindSpore Lite推理的部分对象检测模型的数据。

> 下表的性能是在mate30手机上测试的。

| 模型名称               | 模型链接 | 大小 | 精度 | CPU 4线程时延 |
|-----------------------|----------|----------|----------|-----------|
| SSD |         |         |         |          |
| Faster_RCNN |          |         |         |          |
| YoloV3_Darknet53 | |  |  |  |
| Mask_RCNN | |  |  |  |

