# 比较与tf.keras.datasets.cifar100的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/cifar100.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## tf.keras.datasets.cifar100

```python
class tf.keras.datasets.cifar100()
```

更多内容详见[tf.keras.datasets.cifar100](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/cifar100)。

## mindspore.dataset.Cifar100Dataset

```python
class mindspore.dataset.Cifar100Dataset(
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

更多内容详见[mindspore.dataset.Cifar100Dataset](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset/mindspore.dataset.Cifar100Dataset.html#mindspore.dataset.Cifar100Dataset)。

## 使用方式

TensorFlow：可使用类内的 `load_data` 方法下载并加载Cifar100数据集。

MindSpore：从指定路径加载Cifar100数据集文件，并返回数据集对象。

## 代码示例

```python
# The following implements Cifar100Dataset with MindSpore.
import mindspore.dataset as ds

cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir)

# The following implements cifar100 with TensorFlow.
import tensorflow as tf

x_train, y_train, x_test, y_test = tf.keras.datasets.cifar100.load_data()
```
