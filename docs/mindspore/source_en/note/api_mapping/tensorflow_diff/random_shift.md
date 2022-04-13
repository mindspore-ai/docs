# Function Differences with tf.keras.preprocessing.image.random_shift

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/random_shift.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.preprocessing.image.random_shift](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/preprocessing/image/random_shift).

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

For more information, see [mindspore.dataset.vision.c_transforms.RandomAffine](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomAffine.html#mindspore.dataset.vision.c_transforms.RandomAffine).

## Differences

TensorFlow: Randomly shift the image. The index of axis for rows, columns and channels can be specified by input parameters.

MindSpore: Perform random affine transformation on the image, including random shift. The image needs to be arranged in the order of rows, columns, and channels.

## Code Example

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
