# 比较与tf.image.central_crop的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/central_crop.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.central_crop

```python
tf.image.central_crop(
    image,
    central_fraction
)
```

更多内容详见[tf.image.central_crop](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/central_crop)。

## mindspore.dataset.vision.c_transforms.CenterCrop

```python
class mindspore.dataset.vision.c_transforms.CenterCrop(
    size
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.CenterCrop](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.CenterCrop.html#mindspore.dataset.vision.c_transforms.CenterCrop)。

## 使用方式

TensorFlow：对图像进行中心裁剪，需要输入期望的裁切比例。

MindSpore：对图像进行中心裁剪，需要输入期望的裁切大小。

## 代码示例

```python
# The following implements CenterCrop with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.CenterCrop((14, 14))(image)
print(result.shape)
# (14, 14, 3)

# The following implements central_crop with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.central_crop(image, 0.5)
print(result.shape)
# (14, 14, 3)
```
