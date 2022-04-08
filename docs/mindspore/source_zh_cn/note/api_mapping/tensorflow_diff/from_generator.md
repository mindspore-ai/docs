# 比较与tf.data.Dataset.from_generator的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/from_generator.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.data.Dataset.from_generator

```python
@staticmethod
tf.data.Dataset.from_generator(
    generator,
    output_types,
    output_shapes=None,
    args=None
)
```

更多内容详见[tf.data.Dataset.from_generator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#from_generator)。

## mindspore.dataset.GeneratorDataset

```python
class mindspore.dataset.GeneratorDataset(
    source,
    column_names=None,
    column_types=None,
    schema=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    python_multiprocessing=True,
    max_rowsize=6
)
```

更多内容详见[mindspore.dataset.GeneratorDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)。

## 使用方式

TensorFlow：一个静态方法，支持从可调用对象中创建数据集，并指定数据的类型和形状。

MindSpore：一个数据集类，支持从可调用对象、可迭代对象或可随机访问对象中创建数据集，通过 `schema` 指定数据的类型和形状。

## 代码示例

```python
# The following implements GeneratorDataset with MindSpore.
import numpy as np
import mindspore.dataset as ds

def gen():
    for i in range(1, 3):
        yield np.array([i]), np.array([1] * i)

dataset = ds.GeneratorDataset(source=gen, column_names=["col1", "col2"])

for item in dataset.create_dict_iterator():
    print(item["col1"], item["col2"])
# [1] [1]
# [2] [1 1]

# The following implements from_generator with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def gen():
    for i in range(1, 3):
        yield i, [1] * i

dataset = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

for value in dataset:
    print(value)
# (1, array([1]))
# (2, array([1, 1]))
```
