# 比较与tf.keras.datasets.fashion_mnist的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/fashion_mnist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.datasets.fashion_mnist

```python
class tf.keras.datasets.fashion_mnist()
```

更多内容详见[tf.keras.datasets.fashion_mnist](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/fashion_mnist)。

## mindspore.dataset.FashionMnistDataset

```python
class mindspore.dataset.FashionMnistDataset(
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

更多内容详见[mindspore.dataset.FashionMnistDataset](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset/mindspore.dataset.FashionMnistDataset.html#mindspore.dataset.FashionMnistDataset)。

## 使用方式

TensorFlow：可使用类内的 `load_data` 方法下载并加载Fashion MNIST数据集。

MindSpore：从指定路径加载Fashion MNIST数据集文件，并返回数据集对象。

## 代码示例

```python
# The following implements FashionMnistDataset with MindSpore.
import mindspore.dataset as ds

fashion_mnist_dataset_dir = "/path/to/fashion_mnist_dataset_directory"
dataset = ds.FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir)

# The following implements fashion_mnist with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```
