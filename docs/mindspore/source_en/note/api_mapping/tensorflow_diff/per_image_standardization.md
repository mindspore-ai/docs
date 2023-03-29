# Function Differences with tf.image.per_image_standardization

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/per_image_standardization.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## tf.image.per_image_standardization

```python
tf.image.per_image_standardization(
    image
)
```

For more information, see [tf.image.per_image_standardization](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/per_image_standardization).

## mindspore.dataset.vision.Normalize

```python
class mindspore.dataset.vision.Normalize(
    mean,
    std,
    is_hwc
)
```

For more information, see [mindspore.dataset.vision.Normalize](https://mindspore.cn/docs/en/r2.0/api_python/dataset_vision/mindspore.dataset.vision.Normalize.html#mindspore.dataset.vision.Normalize).

## Differences

TensorFlow: Normalize the image using mean and standard deviation calculated automatically from the image.

MindSpore: Normalize the image using the specified mean and standard deviation.

## Code Example

```python
# The following implements Normalize with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
mean = [np.mean(image, axis=(-1, -2, -3), keepdims=False)]
std = [np.std(image, axis=(-1, -2, -3), keepdims=False)]
adjusted_stddev = list(np.maximum(std, 1.0 / np.sqrt(image.size)))
result = ds.vision.Normalize(mean, adjusted_stddev)(image)
print(result.mean())
# 0.0
print(result.std())
# 1.0

# The following implements per_image_standardization with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

image = tf.random.normal((28, 28, 3))
result = tf.image.per_image_standardization(image)
print(tf.math.reduce_mean(result))
# 0.0
print(tf.math.reduce_std(result))
# 1.0
```
