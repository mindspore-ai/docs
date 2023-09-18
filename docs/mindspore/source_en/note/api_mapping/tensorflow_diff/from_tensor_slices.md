# Function Differences with tf.data.Dataset.from_tensor_slices

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/from_tensor_slices.md)

## tf.data.Dataset.from_tensor_slices

```python
@staticmethod
tf.data.Dataset.from_tensor_slices(
    tensors
)
```

For more information, see [tf.data.Dataset.from_tensor_slices](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/data/Dataset#from_tensor_slices).

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

For more information, see [mindspore.dataset.NumpySlicesDataset](https://www.mindspore.cn/docs/en/r2.1/api_python/dataset/mindspore.dataset.NumpySlicesDataset.html#mindspore.dataset.NumpySlicesDataset).

## Differences

TensorFlow: A static method that creates Dataset with the specified `tf.Tensor`.

MindSpore: A dataset class that creates Dataset with the specified `list` , `tuple` , `dict` or `numpy.ndarray`.

## Code Example

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
