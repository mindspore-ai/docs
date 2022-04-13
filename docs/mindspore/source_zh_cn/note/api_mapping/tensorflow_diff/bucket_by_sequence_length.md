# 比较与tf.data.experimental.bucket_by_sequence_length的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/bucket_by_sequence_length.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.data.experimental.bucket_by_sequence_length

```python
tf.data.experimental.bucket_by_sequence_length(
    element_length_func,
    bucket_boundaries,
    bucket_batch_sizes,
    padded_shapes=None,
    padding_values=None,
    pad_to_bucket_boundary=False,
    no_padding=False,
    drop_remainder=False
)
```

更多内容详见[tf.data.experimental.bucket_by_sequence_length](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/experimental/bucket_by_sequence_length)。

## mindspore.dataset.GeneratorDataset.bucket_batch_by_length

```python
mindspore.dataset.GeneratorDataset.bucket_batch_by_length(
    column_names,
    bucket_boundaries,
    bucket_batch_sizes,
    element_length_function=None,
    pad_info=None,
    pad_to_bucket_boundary=False,
    drop_remainder=False
)
```

更多内容详见[mindspore.dataset.GeneratorDataset.bucket_batch_by_length](https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.bucket_batch_by_length)。

## 使用方式

TensorFlow：根据长度将数据分组，并执行填充和分批，填充形状和填充值由 `padded_shapes` 和 `padding_values` 分别指定。

MindSpore：根据长度将数据分组，并执行填充和分批，填充形状和填充值由 `pad_info` 指定。

## 代码示例

```python
# The following implements bucket_batch_by_length with MindSpore.
import numpy as np
import mindspore.dataset as ds

elements = [
    [0], [1, 2, 3, 4], [5, 6, 7],
    [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

def gen():
    for item in elements:
        yield (np.array([item]),)

dataset = ds.GeneratorDataset(gen, column_names=["data"], shuffle=False)
dataset = dataset.bucket_batch_by_length(
    column_names=["data"],
    bucket_boundaries=[3, 5],
    bucket_batch_sizes=[2, 2, 2],
    element_length_function=lambda elem: elem[0].shape[0])
for item in dataset.create_dict_iterator():
    print(item["data"])
# [[[1 2 3 4]]
#  [[5 6 7 0]]]
# [[[ 7  8  9 10 11  0]]
#  [[13 14 15 16 19 20]]]
# [[[ 0  0]]
#  [[21 22]]]

# The following implements bucket_by_sequence_length with TensorFlow.
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

elements = [
    [0], [1, 2, 3, 4], [5, 6, 7],
    [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

dataset = tf.data.Dataset.from_generator(
    lambda: elements, tf.int64, output_shapes=[None])
dataset = dataset.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda elem: tf.shape(elem)[0],
        bucket_boundaries=[3, 5],
        bucket_batch_sizes=[2, 2, 2]))
for value in dataset.take(3):
    print(value)
# [[1 2 3 4]
#  [5 6 7 0]]
# [[ 7  8  9 10 11  0]
#  [13 14 15 16 19 20]]
# [[ 0  0]
#  [21 22]]
```
