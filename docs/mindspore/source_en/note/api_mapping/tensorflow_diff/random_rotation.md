# Function Differences with tf.keras.preprocessing.image.random_rotation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/random_rotation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.preprocessing.image.random_rotation](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/preprocessing/image/random_rotation).

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

For more information, see [mindspore.dataset.vision.c_transforms.RandomRotation](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomRotation.html#mindspore.dataset.vision.c_transforms.RandomRotation).

## Differences

TensorFlow: Rotate the image with a random degree. The index of axis for rows, columns and channels can be specified by input parameters.

MindSpore: Rotate the image with a random degree and pad the area outside the rotated image. The image needs to be arranged in the order of rows, columns, and channels.

## Code Example

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
