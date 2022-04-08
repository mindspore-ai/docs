# 比较与tf.keras.preprocessing.image.random_shift的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/random_shift.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.preprocessing.image.random_shift

```python
tf.keras.preprocessing.image.random_shift(
    x,
    wrg,
    hrg,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode='nearest',
    cval=0.0,
    interpolation_order=1
)
```

更多内容详见[tf.keras.preprocessing.image.random_shift](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/preprocessing/image/random_shift)。

## mindspore.dataset.vision.c_transforms.RandomAffine

```python
class mindspore.dataset.vision.c_transforms.RandomAffine(
    degrees,
    translate=None,
    scale=None,
    shear=None,
    resample=Inter.NEAREST,
    fill_value=0
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.RandomAffine](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomAffine.html#mindspore.dataset.vision.c_transforms.RandomAffine)。

## 使用方式

TensorFlow：对图像进行随机平移，图像的行、列及通道轴索引可通过入参指定。

MindSpore：对图像进行随机仿射变换，其中包括随机平移，图像需按照行、列、通道的轴顺序排列。

## 代码示例

```python
# The following implements RandomAffine with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.RandomAffine(0, translate=(0.2, 0.3), resample=Inter.NEAREST)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_shift with TensorFlow.
import tensorflow as tf

image = np.random.random((28, 28, 3))
result = tf.keras.preprocessing.image.random_shift(
    image, wrg=0.2, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
print(result.shape)
# (28, 28, 3)
```
