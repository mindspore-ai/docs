# Function Differences with tf.image.convert_image_dtype

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/convert_image_dtype.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.image.convert_image_dtype

```python
tf.image.convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None
)
```

For more information, see [tf.image.convert_image_dtype](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/convert_image_dtype).

## mindspore.dataset.transforms.TypeCast

```python
class mindspore.dataset.transforms.TypeCast(
    output_type
)
```

For more information, see [mindspore.dataset.transforms.TypeCast](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/dataset_transforms/mindspore.dataset.transforms.TypeCast.html#mindspore.dataset.transforms.TypeCast).

## Differences

TensorFlow: Convert the data type of the Tensor image. It supports setting whether to perform clipping before casting to avoid overflow.

MindSpore: Convert the data type of the numpy.ndarray image.

## Code Example

```python
# The following implements TypeCast with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.transforms.TypeCast(np.uint8)(image)
print(result.dtype)
# uint8

# The following implements convert_image_dtype with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3), dtype=tf.float32)
result = tf.image.convert_image_dtype(image, tf.uint8)
print(result.dtype)
# uint8
```
