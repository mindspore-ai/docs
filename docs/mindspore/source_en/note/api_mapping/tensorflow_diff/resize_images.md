# Function Differences with tf.image.resize_images

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/resize_images.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.image.resize_images](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/resize_images).

## mindspore.dataset.vision.c_transforms.Resize

```python
class mindspore.dataset.vision.c_transforms.Resize(
    size,
    interpolation=Inter.LINEAR
)
```

For more information, see [mindspore.dataset.vision.c_transforms.Resize](https://mindspore.cn/docs/en/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Resize.html#mindspore.dataset.vision.c_transforms.Resize).

## Differences

TensorFlow: Resize the image to the specified size. It supports aligning the centers of the 4 corner pixels and preserving the aspect ratio.

MindSpore: Resize the image to the specified size. It will keep the aspect ratio when the input `size` is a single integer.

## Code Example

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
