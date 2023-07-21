# Function Differences with tf.keras.datasets.cifar10

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/cifar10.md)

## tf.keras.datasets.cifar10

```python
class tf.keras.datasets.cifar10()
```

For more information, see [tf.keras.datasets.cifar10](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/cifar10).

## mindspore.dataset.Cifar10Dataset

```python
class mindspore.dataset.Cifar10Dataset(
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

For more information, see [mindspore.dataset.Cifar10Dataset](https://mindspore.cn/docs/en/r2.0/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset).

## Differences

TensorFlow: The Cifar10 dataset can be downloaded and loaded using the `load_data` method within this class.

MindSpore: Load the Cifar10 dataset file from the specified path and return the Dataset.

## Code Example

```python
# The following implements Cifar10Dataset with MindSpore.
import mindspore.dataset as ds

cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir)

# The following implements cifar10 with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```
