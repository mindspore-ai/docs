# 比较与tf.image.resize的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/resize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.resize

```python
tf.image.resize(
    images,
    size,
    method=ResizeMethodV1.BILINEAR,
    align_corners=False,
    preserve_aspect_ratio=False,
    name=None
)
```

更多内容详见[tf.image.resize](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/resize)。

## mindspore.dataset.vision.Resize

```python
class mindspore.dataset.vision.Resize(
    size,
    interpolation=Inter.LINEAR
)
```

更多内容详见[mindspore.dataset.vision.Resize](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/dataset_vision/mindspore.dataset.vision.Resize.html#mindspore.dataset.vision.Resize)。

## 使用方式

TensorFlow：放缩图像至指定大小，支持保留顶角像素点和原始宽高比。

MindSpore：放缩图像至指定大小，当 `size` 输入为单个整数时，将保持原始宽高比。

## 代码示例

```python
# The following implements Resize with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter

image = np.random.random((28, 28, 3))
result = ds.vision.Resize((14, 14), Inter.BICUBIC)(image)
print(result.shape)
# (14, 14, 3)

# The following implements resize with TensorFlow.
import tensorflow as tf
from tensorflow.image import ResizeMethod

image = tf.random.normal((28, 28, 3))
result = tf.image.resize(image, (14, 14), ResizeMethod.BICUBIC)
print(result.shape)
# (14, 14, 3)
```
