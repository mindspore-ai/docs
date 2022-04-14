# 比较与tf.image.pad_to_bounding_box的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/pad_to_bounding_box.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.image.pad_to_bounding_box](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/pad_to_bounding_box)。

## mindspore.dataset.vision.c_transforms.Pad

```python
class mindspore.dataset.vision.c_transforms.Pad(
    padding,
    fill_value=0,
    padding_mode=Border.CONSTANT
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Pad](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Pad.html#mindspore.dataset.vision.c_transforms.Pad)。

## 使用方式

TensorFlow：对图像各边进行填充，入参为上、左边框像素填充数和预期输出图像高、宽，像素填充值为零。

MindSpore：对图像各边进行填充，入参为各边像素填充数、填充值和填充模式。

## 代码示例

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
