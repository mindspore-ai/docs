# 比较与tf.image.random_flip_left_right的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/random_flip_left_right.md)

## tf.image.random_flip_left_right

```python
tf.image.random_flip_left_right(
    image,
    seed=None
)
```

更多内容详见[tf.image.random_flip_left_right](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_flip_left_right)。

## mindspore.dataset.vision.RandomHorizontalFlip

```python
class mindspore.dataset.vision.RandomHorizontalFlip(
    prob=0.5
)
```

更多内容详见[mindspore.dataset.vision.RandomHorizontalFlip](https://mindspore.cn/docs/zh-CN/r1.9/api_python/dataset_vision/mindspore.dataset.vision.RandomHorizontalFlip.html#mindspore.dataset.vision.RandomHorizontalFlip)。

## 使用方式

TensorFlow：随机水平翻转图像，概率为0.5，随机种子可通过入参指定。

MindSpore：随机水平翻转图像，概率可通过入参指定，随机种子需通过 `mindspore.dataset.config.set_seed` 全局设置。

## 代码示例

```python
# The following implements RandomHorizontalFlip with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.RandomHorizontalFlip(prob=0.5)(image)
print(result.shape)
# (28, 28, 3)

# The following implements random_flip_left_right with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_flip_left_right(image, seed=57)
print(result.shape)
# (28, 28, 3)
```
