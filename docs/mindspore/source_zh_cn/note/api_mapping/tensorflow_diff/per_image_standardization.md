# 比较与tf.image.per_image_standardization的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/per_image_standardization.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.per_image_standardization

```python
tf.image.per_image_standardization(
    image
)
```

更多内容详见[tf.image.per_image_standardization](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/per_image_standardization)。

## mindspore.dataset.vision.c_transforms.Normalize

```python
class mindspore.dataset.vision.c_transforms.Normalize(
    mean,
    std
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Normalize](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Normalize.html#mindspore.dataset.vision.c_transforms.Normalize)。

## 使用方式

TensorFlow：对图像进行标准化，均值和标准差将根据图像自动计算。

MindSpore：对图像进行标准化，均值和标准差通过参数输入。

## 代码示例

```python
# The following implements Normalize with MindSpore.
import numpy as np
import mindspore.dataset as ds

image = np.random.random((28, 28, 3))
mean = list(np.mean(image, axis=(-1, -2, -3), keepdims=True))
std = np.std(image, axis=(-1, -2, -3), keepdims=True)
adjusted_stddev = list(np.maximum(std, 1.0 / np.sqrt(image.size)))
result = ds.vision.c_transforms.Normalize(mean, adjusted_stddev)(image)
print(result.mean())
# 0.0
print(result.std())
# 1.0

# The following implements per_image_standardization with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

image = tf.random.normal((28, 28, 3))
result = tf.image.per_image_standardization(image)
print(tf.math.reduce_mean(result))
# 0.0
print(tf.math.reduce_std(result))
# 1.0
```
