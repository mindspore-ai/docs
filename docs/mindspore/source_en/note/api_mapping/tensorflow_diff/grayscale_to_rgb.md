# Function Differences with tf.image.grayscale_to_rgb

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/grayscale_to_rgb.md)

## tf.image.grayscale_to_rgb

```python
tf.image.grayscale_to_rgb(
    images,
    name=None
)
```

For more information, see [tf.image.grayscale_to_rgb](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/grayscale_to_rgb).

## mindspore.dataset.vision.c_transforms.ConvertColor

```python
class mindspore.dataset.vision.c_transforms.ConvertColor(
    convert_mode
)
```

For more information, see [mindspore.dataset.vision.c_transforms.ConvertColor](https://mindspore.cn/docs/en/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.ConvertColor.html#mindspore.dataset.vision.c_transforms.ConvertColor).

## Differences

TensorFlow: Convert the image from grayscale to RGB.

MindSpore: Convert the color space of the image, including conversion from grayscale to RGB.

## Code Example

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
