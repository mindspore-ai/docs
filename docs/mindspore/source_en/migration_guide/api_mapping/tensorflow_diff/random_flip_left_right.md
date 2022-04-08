# Function Differences with tf.image.random_flip_left_right

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/random_flip_left_right.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.image.random_flip_left_right

```python
tf.image.random_flip_left_right(
    image,
    seed=None
)
```

For more information, see [tf.image.random_flip_left_right](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_flip_left_right).

## mindspore.dataset.vision.c_transforms.RandomHorizontalFlip

```python
class mindspore.dataset.vision.c_transforms.RandomHorizontalFlip(
    prob=0.5
)
```

For more information, see [mindspore.dataset.vision.c_transforms.RandomHorizontalFlip](https://mindspore.cn/docs/api/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomHorizontalFlip.html#mindspore.dataset.vision.c_transforms.RandomHorizontalFlip).

## Differences

TensorFlow: Randomly flip the image horizontally with a probability of 0.5. The random seed can be specified by the input parameter.

MindSpore: Randomly flip the image horizontally with the specified probability. The global random seed can to be set through `mindspore.dataset.config.set_seed`.

## Code Example

```python
# The following implements RandomHorizontalFlip with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_flip_left_right with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_flip_left_right(image, seed=57)
print(result.shape)
# (28, 28, 3)
```
