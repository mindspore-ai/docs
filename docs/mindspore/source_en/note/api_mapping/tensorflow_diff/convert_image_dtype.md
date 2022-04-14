# Function Differences with tf.image.convert_image_dtype

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/convert_image_dtype.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## tf.image.convert_image_dtype

```python
tf.image.convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None
)
```

For more information, see [tf.image.convert_image_dtype](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/convert_image_dtype).

## mindspore.dataset.vision.py_transforms.ToType

```python
class mindspore.dataset.vision.py_transforms.ToType(
    output_type
)
```

For more information, see [mindspore.dataset.vision.py_transforms.ToType](https://mindspore.cn/docs/en/r1.7/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToType.html#mindspore.dataset.vision.py_transforms.ToType).

## Differences

TensorFlow: Convert the data type of the Tensor image. It supports setting whether to perform clipping before casting to avoid overflow.

MindSpore: Convert the data type of the numpy.ndarray image.

## Code Example

```python
# The following implements ToType with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.vision.py_transforms.ToType(np.uint8)(image)
print(result.dtype)
# uint8

# The following implements convert_image_dtype with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3), dtype=tf.float32)
result = tf.image.convert_image_dtype(image, tf.uint8)
print(result.dtype)
# uint8
```
