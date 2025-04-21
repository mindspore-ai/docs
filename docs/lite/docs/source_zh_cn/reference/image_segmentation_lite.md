# 图像分割模型

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/reference/image_segmentation_lite.md)

## 图像分割介绍

图像分割是用于检测目标在图片中的位置或者图片中某一像素是输入何种对象的。

使用MindSpore Lite实现图像分割的[示例代码](https://gitee.com/mindspore/models/tree/master/official/lite/image_segmentation)。

## 图像分割模型列表

下表是使用MindSpore Lite推理的部分图像分割模型的数据。

> 下表的性能是在mate30手机上测试的。

| 模型名称               | 大小(Mb) | IoU | CPU 4线程时延(ms) |
|-----------------------| :------: | :-------: | :------: |
| [Deeplabv3](https://download.mindspore.cn/model_zoo/official/lite/deeplabv3_lite/deeplabv3.ms) | 18.7 | 0.58 | 120 |
