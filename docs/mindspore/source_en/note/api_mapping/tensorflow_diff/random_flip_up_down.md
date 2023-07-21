# Function Differences with tf.image.random_flip_up_down

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/random_flip_up_down.md)

## tf.image.random_flip_up_down

```python
tf.image.random_flip_up_down(
    image,
    seed=None
)
```

For more information, see [tf.image.random_flip_up_down](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/random_flip_up_down).

## mindspore.dataset.vision.RandomVerticalFlip

```python
class mindspore.dataset.vision.RandomVerticalFlip(
    prob=0.5
)
```

For more information, see [mindspore.dataset.vision.RandomVerticalFlip](https://mindspore.cn/docs/en/r2.1/api_python/dataset_vision/mindspore.dataset.vision.RandomVerticalFlip.html#mindspore.dataset.vision.RandomVerticalFlip).

## Differences

TensorFlow: Randomly flip the image vertically with a probability of 0.5. The random seed can be specified by the input parameter.

MindSpore: Randomly flip the image vertically with the specified probability. The global random seed can to be set through `mindspore.dataset.config.set_seed`.

## Code Example

```python
# The following implements RandomVerticalFlip with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.RandomVerticalFlip(prob=0.5)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_flip_up_down with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_flip_up_down(image, seed=57)
print(result.shape)
# (28, 28, 3)
```
