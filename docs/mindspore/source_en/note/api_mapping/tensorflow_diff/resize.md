# Function Differences with tf.image.resize

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/resize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.image.resize

```python
tf.image.resize(
    images,
    size,
    method=ResizeMethodV1.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
```

For more information, see [tf.image.resize](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/resize).

## mindspore.dataset.vision.Resize

```python
class mindspore.dataset.vision.Resize(
    size,
    interpolation=Inter.LINEAR
)
```

For more information, see [mindspore.dataset.vision.Resize](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/dataset_vision/mindspore.dataset.vision.Resize.html#mindspore.dataset.vision.Resize).

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
