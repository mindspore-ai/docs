# 比较与tf.data.Dataset.shuffle的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/shuffle.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## tf.data.Dataset.shuffle

```python
tf.data.Dataset.shuffle(
    buffer_size,
    seed=None,
    reshuffle_each_iteration=None
)
```

更多内容详见[tf.data.Dataset.shuffle](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#shuffle)。

## mindspore.dataset.GeneratorDataset.shuffle

```python
mindspore.dataset.GeneratorDataset.shuffle(
    buffer_size
)
```

更多内容详见[mindspore.dataset.GeneratorDataset.shuffle](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.shuffle)。

## 使用方式

TensorFlow：对管道中的数据进行随机混洗，支持设置随机种子和是否在每个迭代中重新混洗。

MindSpore：对管道中的数据进行随机混洗，随机种子需通过 `mindspore.dataset.config.set_seed` 全局设置，且每个迭代都会重新混洗。

## 代码示例

```python
# The following implements shuffle with MindSpore.
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(57)
data = np.array([[1, 2], [3, 4], [5, 6]])
dataset = ds.NumpySlicesDataset(data=data, column_names=["data"], shuffle=False)
dataset = dataset.shuffle(2)

for item in dataset.create_dict_iterator():
    print(item["data"])
# [1 2]
# [5 6]
# [3 4]

# The following implements shuffle with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

data = tf.constant([[1, 2], [3, 4], [5, 6]])
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(2, seed=57)

for value in dataset.take(3):
    print(value)
# [3 4]
# [5 6]
# [1 2]
```
