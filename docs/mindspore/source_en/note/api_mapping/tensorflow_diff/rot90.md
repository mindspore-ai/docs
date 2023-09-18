# Function Differences with tf.image.rot90

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/rot90.md)

## tf.image.rot90

```python
tf.image.rot90(
    image,
    k=1,
    name=None
)
```

For more information, see [tf.image.rot90](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/rot90).

## mindspore.dataset.vision.Rotate

```python
class mindspore.dataset.vision.Rotate(
    degrees,
    resample=Inter.NEAREST,
    expand=False,
    center=None,
    fill_value=0
)
```

For more information, see [mindspore.dataset.vision.Rotate](https://mindspore.cn/docs/en/r2.1/api_python/dataset_vision/mindspore.dataset.vision.Rotate.html#mindspore.dataset.vision.Rotate).

## Differences

TensorFlow: Rotate the image counter-clockwise, by 90 degrees each time.

MindSpore: Rotate the image counter-clockwise with any specified angle, and pad the area outside the rotated image.

## Code Example

```python
# The following implements Rotate with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.array([[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]]])
result = ds.vision.Rotate(90)(image)
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
