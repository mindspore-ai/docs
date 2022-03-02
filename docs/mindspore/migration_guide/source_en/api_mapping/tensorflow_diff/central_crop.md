# Function Differences with tf.image.central_crop

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/central_crop.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## tf.image.central_crop

```python
tf.image.central_crop(
    image,
    central_fraction
)
```

For more information, see [tf.image.central_crop](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/central_crop).

## mindspore.dataset.vision.c_transforms.CenterCrop

```python
class mindspore.dataset.vision.c_transforms.CenterCrop(
    size
)
```

For more information, see [mindspore.dataset.vision.c_transforms.CenterCrop](https://mindspore.cn/docs/api/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.CenterCrop.html#mindspore.dataset.vision.c_transforms.CenterCrop).

## Differences

TensorFlow: Crop the central region of the image with desired crop ratio.

MindSpore: Crop the central region of the image with desired output size.

## Code Example

```python
# The following implements CenterCrop with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.CenterCrop((14, 14))(image)
print(result.shape)
# (14, 14, 3)

# The following implements central_crop with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.central_crop(image, 0.5)
print(result.shape)
# (14, 14, 3)
```
