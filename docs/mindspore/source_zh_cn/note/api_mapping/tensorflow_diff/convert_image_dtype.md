# 比较与tf.image.convert_image_dtype的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/convert_image_dtype.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.convert_image_dtype

```python
tf.image.convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None
)
```

更多内容详见[tf.image.convert_image_dtype](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/image/convert_image_dtype)。

## mindspore.dataset.transforms.TypeCast

```python
class mindspore.dataset.transforms.TypeCast(
    output_type
)
```

更多内容详见[mindspore.dataset.transforms.TypeCast](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_transforms/mindspore.dataset.transforms.TypeCast.html#mindspore.dataset.transforms.TypeCast)。

## 使用方式

TensorFlow：转换Tensor格式图像的数据类型，支持设置是否在转换前进行数值裁切避免溢出。

MindSpore：转换numpy.ndarray格式图像的数据类型。

## 代码示例

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
