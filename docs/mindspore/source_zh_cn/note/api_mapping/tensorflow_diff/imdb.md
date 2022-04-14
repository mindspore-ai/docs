# 比较与tf.keras.datasets.imdb的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/imdb.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## tf.keras.datasets.imdb

```python
class tf.keras.datasets.imdb()
```

更多内容详见[tf.keras.datasets.imdb](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/imdb)。

## mindspore.dataset.IMDBDataset

```python
class mindspore.dataset.IMDBDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None
)
```

更多内容详见[mindspore.dataset.IMDBDataset](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset/mindspore.dataset.IMDBDataset.html#mindspore.dataset.IMDBDataset)。

## 使用方式

TensorFlow：可使用类内的 `load_data` 方法下载并加载IMDB数据集。

MindSpore：从指定路径加载IMDB数据集文件，并返回数据集对象。

## 代码示例

```python
# The following implements IMDBDataset with MindSpore.
import mindspore.dataset as ds

imdb_dataset_dir = "/path/to/imdb_dataset_directory"
dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir)

# The following implements imdb with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
```
