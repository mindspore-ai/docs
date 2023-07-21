# Function Differences with tf.image.crop_to_bounding_box

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/crop_to_bounding_box.md)

## tf.image.crop_to_bounding_box

```python
tf.image.crop_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
)
```

For more information, see [tf.image.crop_to_bounding_box](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/crop_to_bounding_box).

## mindspore.dataset.vision.Crop

```python
class mindspore.dataset.vision.Crop(
    coordinates,
    size
)
```

For more information, see [mindspore.dataset.vision.Crop](https://mindspore.cn/docs/en/r1.10/api_python/dataset_vision/mindspore.dataset.vision.Crop.html#mindspore.dataset.vision.Crop).

## Differences

TensorFlow: Crop at the specified position of the image. Input parameters are the height and width coordinates of the position and the height and width of the cropped image.

MindSpore: Crop at the specified position of the image. Input parameters are the coordinates of the position and the size of the crop image.

## Code Example

```python
# The following implements Crop with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.vision.Crop((0, 0), (14, 14))(image)
print(result.shape)
# (14, 14, 3)

# The following implements crop_to_bounding_box with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.crop_to_bounding_box(image, 0, 0, 14, 14)
print(result.shape)
# (14, 14, 3)
```
