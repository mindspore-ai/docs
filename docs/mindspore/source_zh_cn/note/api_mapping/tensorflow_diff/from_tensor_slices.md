# 比较与tf.data.Dataset.from_tensor_slices的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/from_tensor_slices.md)

## tf.data.Dataset.from_tensor_slices

```python
@staticmethod
tf.data.Dataset.from_tensor_slices(
    tensors
)
```

更多内容详见[tf.data.Dataset.from_tensor_slices](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#from_tensor_slices)。

## mindspore.dataset.NumpySlicesDataset

```python
class mindspore.dataset.NumpySlicesDataset(
    data,
    column_names=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None
)
```

更多内容详见[mindspore.dataset.NumpySlicesDataset](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/dataset/mindspore.dataset.NumpySlicesDataset.html#mindspore.dataset.NumpySlicesDataset)。

## 使用方式

TensorFlow：一个静态方法，使用指定的 `tf.Tensor` 创建数据集。

MindSpore：一个数据集类，使用指定的 `list` 、 `tuple` 、 `dict` 或 `numpy.ndarray` 创建数据集。

## 代码示例

```python
# The following implements NumpySlicesDataset with MindSpore.
import numpy as np
import mindspore.dataset as ds

data = np.array([[1, 2], [3, 4], [5, 6]])
dataset = ds.NumpySlicesDataset(data=data, column_names=["data"], shuffle=False)

for item in dataset.create_dict_iterator():
    print(item["data"])
# [1 2]
# [3 4]
# [5 6]

# The following implements from_tensor_slices with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

data = tf.constant([[1, 2], [3, 4], [5, 6]])
dataset = tf.data.Dataset.from_tensor_slices(data)

for value in dataset:
    print(value)
# [1 2]
# [3 4]
# [5 6]
```
