# 比较与tf.image.rot90的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/rot90.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## tf.image.rot90

```python
tf.image.rot90(
    image,
    k=1,
    name=None
)
```

更多内容详见[tf.image.rot90](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/rot90)。

## mindspore.dataset.vision.c_transforms.Rotate

```python
class mindspore.dataset.vision.c_transforms.Rotate(
    degrees,
    resample=Inter.NEAREST,
    expand=False,
    center=None,
    fill_value=0
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Rotate](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Rotate.html#mindspore.dataset.vision.c_transforms.Rotate)。

## 使用方式

TensorFlow：逆时针旋转图像，每次旋转90度。

MindSpore：逆时针旋转图像，旋转角可为任意度数，并对旋转图像之外的区域进行像素填充。

## 代码示例

```python
# The following implements Rotate with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.array([[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]]])
result = ds.vision.c_transforms.Rotate(90)(image)
print(result)
# [[[0. ], [0. ]],
#  [[0.4], [0.6]],
#  [[0.3], [0.5]]]

# The following implements rot90 with TensorFlow.
import tensorflow as tf

image = tf.constant([[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]]])
result = tf.image.rot90(image, k=1)
print(result)
# [[[0.2], [0.4], [0.6]],
#  [[0.1], [0.3], [0.5]]]
```
