# Function Differences with tf.data.TFRecordDataset

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TFRecordDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.data.TFRecordDataset

```python
class tf.data.TFRecordDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None
)
```

For more information, see [tf.data.TFRecordDataset](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/TFRecordDataset).

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

For more information, see [mindspore.dataset.TFRecordDataset](https://mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.TFRecordDataset.html#mindspore.dataset.TFRecordDataset).

## Differences

TensorFlow: Create Dataset from a list of TFRecord files. It supports decompression operations and can set the cache size.

MindSpore: Create Dataset from a list of TFRecord files. It supports setting the number of samples and the schema of data.

## Code Example

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
