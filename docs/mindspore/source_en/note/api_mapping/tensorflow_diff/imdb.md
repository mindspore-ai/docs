# Function Differences with tf.keras.datasets.imdb

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/imdb.md)

## tf.keras.datasets.imdb

```python
class tf.keras.datasets.imdb()
```

For more information, see [tf.keras.datasets.imdb](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/imdb).

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

For more information, see [mindspore.dataset.IMDBDataset](https://mindspore.cn/docs/en/r2.1/api_python/dataset/mindspore.dataset.IMDBDataset.html#mindspore.dataset.IMDBDataset).

## Differences

TensorFlow: The IMDB dataset can be downloaded and loaded using the `load_data` method within this class.

MindSpore: Load the IMDB dataset file from the specified path and return the Dataset.

## Code Example

```python
# The following implements IMDBDataset with MindSpore.
import mindspore.dataset as ds

imdb_dataset_dir = "/path/to/imdb_dataset_directory"
dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir)

# The following implements imdb with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
```
