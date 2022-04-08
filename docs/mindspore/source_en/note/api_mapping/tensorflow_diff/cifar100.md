# Function Differences with tf.keras.datasets.cifar100

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/cifar100.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.datasets.cifar100

```python
class tf.keras.datasets.cifar100()
```

For more information, see [tf.keras.datasets.cifar100](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/cifar100).

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

For more information, see [mindspore.dataset.Cifar100Dataset](https://mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.Cifar100Dataset.html#mindspore.dataset.Cifar100Dataset).

## Differences

TensorFlow: The Cifar100 dataset can be downloaded and loaded using the `load_data` method within this class.

MindSpore: Load the Cifar100 dataset file from the specified path and return the Dataset.

## Code Example

```python
# The following implements Cifar100Dataset with MindSpore.
import mindspore.dataset as ds

cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir)

# The following implements cifar100 with TensorFlow.
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
```
