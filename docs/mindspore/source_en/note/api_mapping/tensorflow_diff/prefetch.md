# Function Differences with tf.data.Dataset.prefetch

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/prefetch.md)

## tf.data.Dataset.prefetch

```python
tf.data.Dataset.prefetch(
    buffer_size
)
```

For more information, see [tf.data.Dataset.prefetch](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#prefetch).

## mindspore.dataset.config.set_prefetch_size

```python
mindspore.dataset.config.set_prefetch_size(
    size
)
```

For more information, see [mindspore.dataset.config.set_prefetch_size](https://mindspore.cn/docs/en/r1.7/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size).

## Differences

TensorFlow: A method of the `Dataset` class, used to set the size of the current data pipeline cache queue.

MindSpore: A function to set the global size of all data pipeline cache queues.

## Code Example

```python
# The following implements set_prefetch_size with MindSpore.
import mindspore.dataset as ds

ds.config.set_prefetch_size(2)

# The following implements prefetch with TensorFlow.
import tensorflow as tf

data = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.prefetch(2)
```
