# Function Differences with tf.io.decode_image

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/decode_image.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.io.decode_image

```python
tf.io.decode_image(
    contents,
    channels=None,
    dtype=tf.dtypes.uint8,
    name=None,
    expand_animations=True
)
```

For more information, see [tf.io.decode_image](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/io/decode_image).

## mindspore.dataset.vision.c_transforms.Decode

```python
class mindspore.dataset.vision.c_transforms.Decode(
    rgb=True
)
```

For more information, see [mindspore.dataset.vision.c_transforms.Decode](https://mindspore.cn/docs/api/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Decode.html#mindspore.dataset.vision.c_transforms.Decode).

## Differences

TensorFlow: Decode the raw image bytes into an image with the specified number of channels and data type. It supports decoding GIF images.

MindSpore: Decode the raw image bytes into a RGB image.

## Code Example

```python
# The following implements Decode with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.fromfile("/tmp/file.jpg", dtype=np.uint8)
result = ds.vision.c_transforms.Decode()(image)

# The following implements decode_image with TensorFlow.
import tensorflow as tf

raw = tf.io.read_file("/tmp/file.jpg")
result = tf.io.decode_image(raw, channels=3)
```
