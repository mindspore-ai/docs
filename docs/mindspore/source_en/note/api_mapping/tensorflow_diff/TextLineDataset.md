# Function Differences with tf.data.TextLineDataset

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TextLineDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## tf.data.TextLineDataset

```python
class tf.data.TextLineDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None
)
```

For more information, see [tf.data.TextLineDataset](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/TextLineDataset).

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

For more information, see [mindspore.dataset.TextFileDataset](https://mindspore.cn/docs/en/r1.7/api_python/dataset/mindspore.dataset.TextFileDataset.html#mindspore.dataset.TextFileDataset).

## Differences

TensorFlow: Create Dataset from a list of text files. It supports decompression operations and can set the cache size.

MindSpore: Create Dataset from a list of text files. It supports setting the number of samples.

## Code Example

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
