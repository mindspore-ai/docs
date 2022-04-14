# 比较与tf.keras.datasets.mnist的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/mnist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.datasets.mnist

```python
class tf.keras.datasets.mnist()
```

更多内容详见[tf.keras.datasets.mnist](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/mnist)。

## mindspore.dataset.MnistDataset

```python
class mindspore.dataset.MnistDataset(
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

更多内容详见[mindspore.dataset.MnistDataset](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset)。

## 使用方式

TensorFlow：可使用类内的 `load_data` 方法下载并加载MNIST数据集。

MindSpore：从指定路径加载MNIST数据集文件，并返回数据集对象。

## 代码示例

```python
# The following implements MnistDataset with MindSpore.
import mindspore.dataset as ds

mnist_dataset_dir = "/path/to/mnist_dataset_directory"
dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir)

# The following implements mnist with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
