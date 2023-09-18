# Function Differences with tf.data.experimental.CsvDataset

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/CsvDataset.md)

## tf.data.experimental.CsvDataset

```python
class tf.data.experimental.CsvDataset(
    filenames,
    record_defaults,
    compression_type=None,
    buffer_size=None,
    header=False,
    field_delim=',',
    use_quote_delim=True,
    na_value='',
    select_cols=None,
    exclude_cols=None
)
```

For more information, see [tf.data.experimental.CsvDataset](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/data/experimental/CsvDataset).

## mindspore.dataset.CSVDataset

```python
class mindspore.dataset.CSVDataset(
    dataset_files,
    field_delim=', ',
    column_defaults=None,
    column_names=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None
)
```

For more information, see [mindspore.dataset.CSVDataset](https://www.mindspore.cn/docs/en/r2.1/api_python/dataset/mindspore.dataset.CSVDataset.html#mindspore.dataset.CSVDataset).

## Differences

TensorFlow: Create Dataset from a list of CSV files. It supports decompression operations and can set cache size and skip file headers.

MindSpore: Create Dataset from a list of CSV files. It supports setting the number of samples.

## Code Example

```python
# The following implements CSVDataset with MindSpore.
import mindspore.dataset as ds

dataset_files = ['/tmp/example0.csv',
                 '/tmp/example1.csv']
dataset = ds.TextFileDataset(dataset_files)

# The following implements CsvDataset with TensorFlow.
import tensorflow as tf

filenames = ['/tmp/example0.csv',
             '/tmp/example1.csv']
dataset = tf.data.experimental.CsvDataset(filenames,
                                          [tf.float32,
                                           tf.constant([0.0], dtype=tf.float32),
                                           tf.int32])
```
