# 比较与tf.image.rgb_to_grayscale的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/rgb_to_grayscale.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.rgb_to_grayscale

```python
tf.image.rgb_to_grayscale(
    images,
    name=None
)
```

更多内容详见[tf.image.rgb_to_grayscale](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/rgb_to_grayscale)。

## mindspore.dataset.vision.c_transforms.ConvertColor

```python
class mindspore.dataset.vision.c_transforms.ConvertColor(
    convert_mode
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.ConvertColor](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.ConvertColor.html#mindspore.dataset.vision.c_transforms.ConvertColor)。

## 使用方式

TensorFlow：将图像从RGB图转换为灰度图。

MindSpore：转换图像的色彩空间，其中包括将RGB图转换为灰度图。

## 代码示例

```python
# The following implements ConvertColor with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import ConvertMode

image = np.random.random((28, 28, 3)).astype(np.float32)
result = ds.vision.c_transforms.ConvertColor(ConvertMode.COLOR_RGB2GRAY)(image)
print(result.shape)
# (28, 28)

# The following implements rgb_to_grayscale with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.rgb_to_grayscale(image)
print(result.shape)
# (28, 28, 1)
```
