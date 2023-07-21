# Function Differences with tf.image.random_crop

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/random_crop.md)

## tf.image.random_crop

```python
tf.image.random_crop(
    value,
    size,
    seed=None,
    name=None
)
```

For more information, see [tf.image.random_crop](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_crop).

## mindspore.dataset.vision.RandomCrop

```python
class mindspore.dataset.vision.RandomCrop(
    size,
    padding=None,
    pad_if_needed=False,
    fill_value=0,
    padding_mode=Border.CONSTANT
)
```

For more information, see [mindspore.dataset.vision.RandomCrop](https://mindspore.cn/docs/en/r1.9/api_python/dataset_vision/mindspore.dataset.vision.RandomCrop.html#mindspore.dataset.vision.RandomCrop).

## Differences

TensorFlow: Crop the image at a random position with the specified random seed.

MindSpore: Crop the image at a random position and pad if needed. The global random seed can be set through `mindspore.dataset.config.set_seed`.

## Code Example

```python
# The following implements RandomCrop with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.RandomCrop((5, 5))(image)
print(result.shape)
# (5, 5, 3)

# The following implements random_crop with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_crop(image, (5, 5, 3), seed=57)
print(result.shape)
# (5, 5, 3)
```
