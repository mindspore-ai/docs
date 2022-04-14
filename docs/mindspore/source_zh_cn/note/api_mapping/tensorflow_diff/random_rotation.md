# 比较与tf.keras.preprocessing.image.random_rotation的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/random_rotation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.preprocessing.image.random_rotation

```python
tf.keras.preprocessing.image.random_rotation(
    x,
    rg,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode='nearest',
    cval=0.0,
    interpolation_order=1
)
```

更多内容详见[tf.keras.preprocessing.image.random_rotation](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/preprocessing/image/random_rotation)。

## mindspore.dataset.vision.c_transforms.RandomRotation

```python
class mindspore.dataset.vision.c_transforms.RandomRotation(
    degrees,
    resample=Inter.NEAREST,
    expand=False,
    center=None,
    fill_value=0
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.RandomRotation](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomRotation.html#mindspore.dataset.vision.c_transforms.RandomRotation)。

## 使用方式

TensorFlow：对图像进行随机旋转，图像的行、列及通道轴索引可通过入参指定。

MindSpore：对图像进行随机旋转，并对旋转图像之外的区域进行像素填充，图像需按照行、列、通道的轴顺序排列。

## 代码示例

```python
# The following implements RandomRotation with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.RandomRotation(90, resample=Inter.NEAREST)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_rotation with TensorFlow.
import tensorflow as tf

image = np.random.random((28, 28, 3))
result = tf.keras.preprocessing.image.random_rotation(
    image, 90, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
print(result.shape)
# (28, 28, 3)
```
