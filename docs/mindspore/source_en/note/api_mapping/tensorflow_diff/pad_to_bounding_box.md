# Function Differences with tf.image.pad_to_bounding_box

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/pad_to_bounding_box.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.image.pad_to_bounding_box

```python
tf.image.pad_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
)
```

For more information, see [tf.image.pad_to_bounding_box](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/pad_to_bounding_box).

## mindspore.dataset.vision.c_transforms.Pad

```python
class mindspore.dataset.vision.c_transforms.Pad(
    padding,
    fill_value=0,
    padding_mode=Border.CONSTANT
)
```

For more information, see [mindspore.dataset.vision.c_transforms.Pad](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Pad.html#mindspore.dataset.vision.c_transforms.Pad).

## Differences

TensorFlow: Pad the borders of the image. Input parameters are the number of rows and columns to add on the top and left border, and the height and width of the desired output image. The pixel padding value is zero.

MindSpore: Pad the borders of the image. Input parameters are the number of pixels to add on each border, the pixel padding value, and the padding mode.

## Code Example

```python
# The following implements Pad with MindSpore.
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import Border

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.Pad((2, 2, 5, 5), 0, Border.CONSTANT)(image)
print(result.shape)
# (35, 35, 3)

# The following implements pad_to_bounding_box with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.pad_to_bounding_box(image, 2, 2, 35, 35)
print(result.shape)
# (35, 35, 3)
```
