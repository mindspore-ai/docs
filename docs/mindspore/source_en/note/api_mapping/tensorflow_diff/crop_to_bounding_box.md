# Function Differences with tf.image.crop_to_bounding_box

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/crop_to_bounding_box.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

## mindspore.dataset.vision.c_transforms.Crop

```python
class mindspore.dataset.vision.c_transforms.Crop(
    coordinates,
    size
)
```

For more information, see [mindspore.dataset.vision.c_transforms.Crop](https://mindspore.cn/docs/api/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Crop.html#mindspore.dataset.vision.c_transforms.Crop).

## Differences

TensorFlow: Crop at the specified position of the image. Input parameters are the height and width coordinates of the position and the height and width of the cropped image.

MindSpore: Crop at the specified position of the image. Input parameters are the coordinates of the position and the size of the crop image.

## Code Example

```python
# The following implements Crop with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
result = ds.vision.c_transforms.Crop((0, 0), (14, 14))(image)
print(result.shape)
# (14, 14, 3)

# The following implements crop_to_bounding_box with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.crop_to_bounding_box(image, 0, 0, 14, 14)
print(result.shape)
# (14, 14, 3)
```
