# Function Differences with tf.keras.datasets.fashion_mnist

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/fashion_mnist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.datasets.fashion_mnist

```python
class tf.keras.datasets.fashion_mnist()
```

For more information, see [tf.keras.datasets.fashion_mnist](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/fashion_mnist).

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

For more information, see [mindspore.dataset.FashionMnistDataset](https://mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.FashionMnistDataset.html#mindspore.dataset.FashionMnistDataset).

## Differences

TensorFlow: The Fashion MNIST dataset can be downloaded and loaded using the `load_data` method within this class.

MindSpore: Load the Fashion MNIST dataset file from the specified path and return the Dataset.

## Code Example

```python
# The following implements FashionMnistDataset with MindSpore.
import mindspore.dataset as ds

fashion_mnist_dataset_dir = "/path/to/fashion_mnist_dataset_directory"
dataset = ds.FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir)

# The following implements fashion_mnist with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```
