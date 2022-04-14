# 比较与tf.image.convert_image_dtype的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/convert_image_dtype.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.convert_image_dtype

```python
tf.image.convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None
)
```

更多内容详见[tf.image.convert_image_dtype](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/convert_image_dtype)。

## mindspore.dataset.vision.py_transforms.ToType

```python
class mindspore.dataset.vision.py_transforms.ToType(
    output_type
)
```

更多内容详见[mindspore.dataset.vision.py_transforms.ToType](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToType.html#mindspore.dataset.vision.py_transforms.ToType)。

## 使用方式

TensorFlow：转换Tensor格式图像的数据类型，支持设置是否在转换前进行数值裁切避免溢出。

MindSpore：转换numpy.ndarray格式图像的数据类型。

## 代码示例

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
