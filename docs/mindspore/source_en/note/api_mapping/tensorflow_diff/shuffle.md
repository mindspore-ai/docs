# Function Differences with tf.data.Dataset.shuffle

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/shuffle.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## tf.data.Dataset.shuffle

```python
tf.data.Dataset.shuffle(
    buffer_size,
    seed=None,
    reshuffle_each_iteration=None
)
```

For more information, see [tf.data.Dataset.shuffle](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#shuffle).

## mindspore.dataset.GeneratorDataset.shuffle

```python
mindspore.dataset.GeneratorDataset.shuffle(
    buffer_size
)
```

For more information, see [mindspore.dataset.GeneratorDataset.shuffle](https://mindspore.cn/docs/en/r1.7/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.shuffle).

## Differences

TensorFlow: Randomly shuffle the data in the pipeline. It supports setting a random seed and whether to reshuffle at each iteration.

MindSpore: Randomly shuffle the data in the pipeline. The global random seed can be set through `mindspore.dataset.config.set_seed`, and it will reshuffe every iteration.

## Code Example

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
