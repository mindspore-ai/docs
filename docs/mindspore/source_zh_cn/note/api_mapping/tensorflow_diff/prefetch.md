# 比较与tf.data.Dataset.prefetch的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/prefetch.md)

## tf.data.Dataset.prefetch

```python
tf.data.Dataset.prefetch(
    buffer_size
)
```

更多内容详见[tf.data.Dataset.prefetch](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset#prefetch)。

## mindspore.dataset.config.set_prefetch_size

```python
mindspore.dataset.config.set_prefetch_size(
    size
)
```

更多内容详见[mindspore.dataset.config.set_prefetch_size](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size)。

## 使用方式

TensorFlow：`Dataset` 类内函数，用于设置当前数据管道缓存队列的大小。

MindSpore：全局配置函数，用于设置所有数据管道缓存队列的大小。

## 代码示例

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
