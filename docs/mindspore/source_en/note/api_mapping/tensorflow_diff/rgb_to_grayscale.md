# Function Differences with tf.image.rgb_to_grayscale

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/rgb_to_grayscale.md)

## tf.image.rgb_to_grayscale

```python
tf.image.rgb_to_grayscale(
    images,
    name=None
)
```

For more information, see [tf.image.rgb_to_grayscale](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/rgb_to_grayscale).

## mindspore.dataset.vision.ConvertColor

```python
class mindspore.dataset.vision.ConvertColor(
    convert_mode
)
```

For more information, see [mindspore.dataset.vision.ConvertColor](https://mindspore.cn/docs/en/r1.10/api_python/dataset_vision/mindspore.dataset.vision.ConvertColor.html#mindspore.dataset.vision.ConvertColor).

## Differences

TensorFlow: Convert the image from RGB to grayscale.

MindSpore: Convert the color space of the image, including conversion from RGB to grayscale.

## Code Example

```python
# The following implements ConvertColor with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import ConvertMode

image = np.random.random((28, 28, 3)).astype(np.float32)
result = ds.vision.ConvertColor(ConvertMode.COLOR_RGB2GRAY)(image)
print(result.shape)
# (28, 28)

# The following implements rgb_to_grayscale with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.rgb_to_grayscale(image)
print(result.shape)
# (28, 28, 1)
```
