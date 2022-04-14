# Function Differences with tf.keras.datasets.mnist

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/mnist.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.datasets.mnist

```python
class tf.keras.datasets.mnist()
```

For more information, see [tf.keras.datasets.mnist](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/mnist).

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

For more information, see [mindspore.dataset.MnistDataset](https://mindspore.cn/docs/en/r1.7/api_python/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset).

## Differences

TensorFlow: The MNIST dataset can be downloaded and loaded using the `load_data` method inside this class.

MindSpore: Load the MNIST dataset file from the specified path and return the Dataset.

## Code Example

```python
# The following implements MnistDataset with MindSpore.
import mindspore.dataset as ds

mnist_dataset_dir = "/path/to/mnist_dataset_directory"
dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir)

# The following implements mnist with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
