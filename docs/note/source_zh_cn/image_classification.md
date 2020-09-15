# 图像分类

<a href="https://gitee.com/mindspore/docs/blob/master/lite/docs/source_zh_cn/image_classification.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 图像分类介绍

图像分类模型可以预测图片中出现哪些物体，识别出图片中出现物体列表及其概率。 比如下图经过模型推理的分类结果为下表： 

![image_classification](images/image_classification_result.png)

| 类别       | 概率   |
| ---------- | ------ |
| plant      | 0.9359 |
| flower     | 0.8641 |
| tree       | 0.8584 |
| houseplant | 0.7867 |

使用MindSpore Lite实现图像分类的[示例代码](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/lite/image_classification)。

## 图像分类模型列表

下表是使用MindSpore Lite推理的部分图像分类模型的数据。

> 下表的性能是在mate30手机上测试的。

| 模型名称               | 模型链接 | 大小 | 精度 | CPU 4线程时延 |
|-----------------------|----------|----------|----------|-----------|
| MobileNetV2 |         |         |         |          |
| LeNet |          |         |         |          |
| AlexNet | |  |  |  |
| GoogleNet | |  |  |  |
| ResNext50 | |  |  |  |

