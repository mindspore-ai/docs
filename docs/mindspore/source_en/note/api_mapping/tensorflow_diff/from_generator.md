# Function Differences with tf.data.Dataset.from_generator

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/from_generator.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## tf.data.Dataset.from_generator

```python
@staticmethod
tf.data.Dataset.from_generator(
    generator,
    output_types=None,
    output_shapes=None,
    args=None,
    output_signature=None
)
```

For more information, see [tf.data.Dataset.from_generator](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/data/Dataset#from_generator).

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

For more information, see [mindspore.dataset.GeneratorDataset](https://www.mindspore.cn/docs/en/r2.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset).

## Differences

TensorFlow: A static method that creates Dataset from a callable object with the specified type and shape.

MindSpore: A dataset class that creates Dataset from a callable, iterable, or random-accessible object with the type and shape specified by `schema`.

## Code Example

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
