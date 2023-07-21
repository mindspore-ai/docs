# 图像分类模型支持（Lite）

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/image_classification_lite.md)

## 图像分类介绍

图像分类模型可以预测图片中出现哪些物体，识别出图片中出现物体列表及其概率。 比如下图经过模型推理的分类结果为下表：

![image_classification](images/image_classification_result.png)

| 类别       | 概率   |
| ---------- | ------ |
| plant      | 0.9359 |
| flower     | 0.8641 |
| tree       | 0.8584 |
| houseplant | 0.7867 |

使用MindSpore Lite实现图像分类的[示例代码](https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo/official/lite/image_classification)。

## 图像分类模型列表

下表是使用MindSpore Lite推理的部分图像分类模型的数据。

> 下表的性能是在mate30手机上测试的。

| 模型名称               | 大小(Mb) | Top1 | Top5 | F1 | CPU 4线程时延(ms) |
|-----------------------| :----------: | :----------: | :----------: | :----------: | :-----------: |
| [MobileNetV2](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms) | 11.5 | - | - | 65.5% | 14.595 |
| [Inceptionv3](https://download.mindspore.cn/model_zoo/official/lite/inceptionv3_lite/inceptionv3.ms) | 90.9 | 78.62% | 94.08% | - | 92.086 |
| [Shufflenetv2](https://download.mindspore.cn/model_zoo/official/lite/shufflenetv2_lite/shufflenetv2.ms) | 8.8 | 67.74% | 87.62% | - | 8.303 |
| [GoogleNet](https://download.mindspore.cn/model_zoo/official/lite/googlenet_lite/googlenet.ms) | 25.3 | 72.2% | 90.06% | - | 23.257 |
| [ResNext50](https://download.mindspore.cn/model_zoo/official/lite/resnext50_lite/resnext50.ms) | 95.8 | 73.1% | 91.21% | - | 138.164 |
