# 比较与tf.data.TFRecordDataset的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/TFRecordDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.data.TFRecordDataset

```python
class tf.data.TFRecordDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None
)
```

更多内容详见[tf.data.TFRecordDataset](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/TFRecordDataset)。

## mindspore.dataset.TFRecordDataset

```python
class mindspore.dataset.TFRecordDataset(
    dataset_files,
    schema=None,
    columns_list=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    shard_equal_rows=False,
    cache=None
)
```

更多内容详见[mindspore.dataset.TFRecordDataset](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset/mindspore.dataset.TFRecordDataset.html#mindspore.dataset.TFRecordDataset)。

## 使用方式

TensorFlow：从TFRecord文件列表创建数据集，支持解压操作，能够设置缓存大小。

MindSpore：从TFRecord文件列表创建数据集，支持设置读取样本的数目以及数据的类型和形状。

## 代码示例

```python
# The following implements TFRecordDataset with MindSpore.
import mindspore.dataset as ds

dataset_files = ['/tmp/example0.tfrecord',
                 '/tmp/example1.tfrecord']
dataset = ds.TFRecordDataset(dataset_files)

# The following implements TFRecordDataset with TensorFlow.
import tensorflow as tf

filenames = ['/tmp/example0.tfrecord',
             '/tmp/example1.tfrecord']
dataset = tf.data.TFRecordDataset(filenames)
```
