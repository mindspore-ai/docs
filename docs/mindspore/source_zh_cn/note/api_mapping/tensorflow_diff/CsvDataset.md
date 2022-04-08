# 比较与tf.data.experimental.CsvDataset的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/CsvDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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
    select_cols=None
)
```

更多内容详见[tf.data.experimental.CsvDataset](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/experimental/CsvDataset)。

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

更多内容详见[mindspore.dataset.CSVDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.CSVDataset.html#mindspore.dataset.CSVDataset)。

## 使用方式

TensorFlow：从CSV文件列表创建数据集，支持解压操作，能够设置缓存大小和跳过文件头。

MindSpore：从CSV文件列表创建数据集，支持设置读取样本的数目。

## 代码示例

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
