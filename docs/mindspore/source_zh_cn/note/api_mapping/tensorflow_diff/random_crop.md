# 比较与tf.image.random_crop的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/random_crop.md)

## tf.image.random_crop

```python
tf.image.random_crop(
    value,
    size,
    seed=None,
    name=None
)
```

更多内容详见[tf.image.random_crop](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/random_crop)。

## mindspore.dataset.vision.RandomCrop

```python
class mindspore.dataset.vision.RandomCrop(
    size,
    padding=None,
    pad_if_needed=False,
    fill_value=0,
    padding_mode=Border.CONSTANT
)
```

更多内容详见[mindspore.dataset.vision.RandomCrop](https://mindspore.cn/docs/zh-CN/r1.8/api_python/dataset_vision/mindspore.dataset.vision.RandomCrop.html#mindspore.dataset.vision.RandomCrop)。

## 使用方式

TensorFlow：对图像进行随机裁剪，随机种子可通过入参指定。

MindSpore：对图像进行随机裁剪，支持同时填充图像，随机种子需通过 `mindspore.dataset.config.set_seed` 全局设置。

## 代码示例

```python
# The following implements RandomCrop with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
image = np.random.random((28, 28, 3))
result = ds.vision.RandomCrop((5, 5))(image)
print(result.shape)
# (5, 5, 3)

# The following implements random_crop with TensorFlow.
import tensorflow as tf

image = tf.random.normal((28, 28, 3))
result = tf.image.random_crop(image, (5, 5, 3), seed=57)
print(result.shape)
# (5, 5, 3)
```
