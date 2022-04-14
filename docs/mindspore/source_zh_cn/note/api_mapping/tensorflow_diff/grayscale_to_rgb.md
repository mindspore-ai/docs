# 比较与tf.image.grayscale_to_rgb的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/grayscale_to_rgb.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## tf.image.grayscale_to_rgb

```python
tf.image.grayscale_to_rgb(
    images,
    name=None
)
```

更多内容详见[tf.image.grayscale_to_rgb](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/grayscale_to_rgb)。

## mindspore.dataset.vision.c_transforms.ConvertColor

```python
class mindspore.dataset.vision.c_transforms.ConvertColor(
    convert_mode
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.ConvertColor](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.ConvertColor.html#mindspore.dataset.vision.c_transforms.ConvertColor)。

## 使用方式

TensorFlow：将图像从灰度图转换为RGB图。

MindSpore：转换图像的色彩空间，其中包括将灰度图转换为RGB图。

## 代码示例

```python
# The following implements ConvertColor with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import ConvertMode

image = np.random.random((28, 28, 1))
result = ds.vision.c_transforms.ConvertColor(ConvertMode.COLOR_GRAY2RGB)(image)
print(result.shape)
# (28, 28, 3)

# The following implements grayscale_to_rgb with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 1))
result = tf.image.grayscale_to_rgb(image)
print(result.shape)
# (28, 28, 3)
```
