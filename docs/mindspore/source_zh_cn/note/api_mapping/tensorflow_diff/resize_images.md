# 比较与tf.image.resize_images的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/resize_images.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.resize_images

```python
tf.image.resize_images(
    images,
    size,
    method=ResizeMethodV1.BILINEAR,
    align_corners=False,
    preserve_aspect_ratio=False,
    name=None
)
```

更多内容详见[tf.image.resize_images](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/resize_images)。

## mindspore.dataset.vision.c_transforms.Resize

```python
class mindspore.dataset.vision.c_transforms.Resize(
    size,
    interpolation=Inter.LINEAR
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Resize](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Resize.html#mindspore.dataset.vision.c_transforms.Resize)。

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
result = ds.vision.c_transforms.Resize((14, 14), Inter.BICUBIC)(image)
print(result.shape)
# (14, 14, 3)

# The following implements resize_images with TensorFlow.
import tensorflow as tf
from tensorflow.image import ResizeMethod

image = tf.random.normal((28, 28, 3))
result = tf.image.resize_images(image, (14, 14), ResizeMethod.BICUBIC)
print(result.shape)
# (14, 14, 3)
```
