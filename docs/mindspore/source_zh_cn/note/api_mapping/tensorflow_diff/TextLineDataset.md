# 比较与tf.data.TextLineDataset的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/TextLineDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.data.TextLineDataset

```python
class tf.data.TextLineDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None
)
```

更多内容详见[tf.data.TextLineDataset](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/TextLineDataset)。

## mindspore.dataset.TextFileDataset

```python
class mindspore.dataset.TextFileDataset(
    dataset_files,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None
)
```

更多内容详见[mindspore.dataset.TextFileDataset](https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.TextFileDataset.html#mindspore.dataset.TextFileDataset)。

## 使用方式

TensorFlow：从文本文件列表创建数据集，支持解压操作，能够设置缓存大小。

MindSpore：从文本文件列表创建数据集，支持设置读取样本的数目。

## 代码示例

```python
# The following implements TextFileDataset with MindSpore.
import mindspore.dataset as ds

dataset_files = ['/tmp/example0.txt',
                 '/tmp/example1.txt']
dataset = ds.TextFileDataset(dataset_files)

# The following implements TextLineDataset with TensorFlow.
import tensorflow as tf

filenames = ['/tmp/example0.txt',
             '/tmp/example1.txt']
dataset = tf.data.TextLineDataset(filenames)
```
