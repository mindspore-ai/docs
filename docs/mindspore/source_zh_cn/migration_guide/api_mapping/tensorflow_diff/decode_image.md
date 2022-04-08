# 比较与tf.io.decode_image的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/decode_image.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.io.decode_image](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/io/decode_image)。

## mindspore.dataset.vision.c_transforms.Decode

```python
class mindspore.dataset.vision.c_transforms.Decode(
    rgb=True
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Decode](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Decode.html#mindspore.dataset.vision.c_transforms.Decode)。

## 使用方式

TensorFlow：将图像字节码解码为指定通道数和数据类型的图像，支持解码动态图。

MindSpore：将图像字节码解码为RGB图像。

## 代码示例

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
