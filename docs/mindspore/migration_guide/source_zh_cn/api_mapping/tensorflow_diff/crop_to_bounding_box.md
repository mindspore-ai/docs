# 比较与tf.image.crop_to_bounding_box的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/crop_to_bounding_box.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.image.crop_to_bounding_box](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/crop_to_bounding_box)。

## mindspore.dataset.vision.c_transforms.Crop

```python
class mindspore.dataset.vision.c_transforms.Crop(
    coordinates,
    size
)
```

更多内容详见[mindspore.dataset.vision.c_transforms.Crop](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Crop.html#mindspore.dataset.vision.c_transforms.Crop)。

## 使用方式

TensorFlow：在图像指定位置进行裁剪，入参为裁剪位置的高、宽坐标和裁剪子图的高、宽。

MindSpore：在图像指定位置进行裁剪，入参为裁剪位置的坐标和裁剪子图的大小。

## 代码示例

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
