[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/dataset.md)

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || **Dataset** || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html)

# Dataset

Data is the foundation of deep learning, and high-quality data input is beneficial to the entire deep neural network. MindSpore provides Pipeline-based [Data Engine](https://www.mindspore.cn/docs/zh-CN/master/design/data_engine.html) and achieves efficient data preprocessing through [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) and [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html), where Dataset is the start of Pipeline and is used to load raw data. `mindspore.dataset` provides built-in dataset interfaces for loading text, image, audio, etc., and provides interfaces for loading customized datasets.

In addition, MindSpore's domain development library also provides a large number of preloaded datasets that can be downloaded and used with one click through the API. This tutorial will elaborate on different dataset loading methods, common dataset operations and customized dataset methods respectively.

```python
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt
```

## Loading a Dataset

We use the **Mnist** dataset as a sample to introduce the loading method by using `mindspore.dataset` .

The interface provided by `mindspore.dataset` **only supports decompressed data files**, so we use the `download` library to download the dataset and decompress it.

```python
# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:02<00:00, 3.96MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

After the compressed file is deleted and loaded directly, you can see that its data type is MnistDataset.

```python
train_dataset = MnistDataset("MNIST_Data/train", shuffle=False)
print(type(train_dataset))
```

```text
<class 'mindspore.dataset.engine.datasets_vision.MnistDataset'>
```

## Iterating a Dataset

After the dataset is loaded, the data is generally acquired in an iterative manner and then fed into the neural network for training. You can use the [create_tuple_iterator](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_tuple_iterator.html) or [create_dict_iterator](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_dict_iterator.html) interface to create a data iterator to iteratively access data. The default type of data to be accessed is `Tensor`. If `output_numpy=True` is set, the type of data to be accessed is `Numpy`.

```python
def visualize(dataset):
    figure = plt.figure(figsize=(4, 4))
    cols, rows = 3, 3

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for idx, (image, label) in enumerate(dataset.create_tuple_iterator()):
        figure.add_subplot(rows, cols, idx + 1)
        plt.title(int(label))
        plt.axis("off")
        plt.imshow(image.asnumpy().squeeze(), cmap="gray")
        if idx == cols * rows - 1:
            break
    plt.show()
```

```python
visualize(train_dataset)
```

## Common Operations on Datasets

The common operations of dataset use the asynchronous execution of `dataset = dataset.operation()` according to The design concept of Pipeline. The execution of the operation returns a new Dataset, at which time no specific operation is executed, but nodes are added to the Pipeline. The whole Pipeline is executed in parallel when iteration is finally performed.

The following are the common operations of datasets.

### shuffle

Random `shuffle` of datasets can eliminate the problem of uneven distribution caused by data alignment.

![op-shuffle](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/dataset/images/op_shuffle.png)

The datasets provided by `mindspore.dataset` can be configured with `shuffle=True` at loading time, or using the following operation:

```python
train_dataset = train_dataset.shuffle(buffer_size=64)

visualize(train_dataset)
```

### map

The `map` is the key operation of data preprocessing, which can add data transforms to a specified column of the dataset, apply data transforms to each element of the column data, and return a new dataset containing the transformed elements.

> For the different types of transforms supported by dataset, see [Data Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html).

```python
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape, image.dtype)
```

```text
(28, 28, 1) UInt8
```

Here data scaling is performed on the Mnist dataset by dividing the image uniformly by 255 and converting the data type from uint8 to float32.

```python
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
```

Comparing the data before and after map, you can see the data type change.

```python
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape, image.dtype)
```

```text
(28, 28, 1) Float32
```

### batch

Packing the dataset into a fixed size `batch` is a compromise method for model optimization using gradient descent with limited hardware resources, which can ensure the randomness of gradient descent and optimize the computational effort.

![op-batch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/dataset/images/op_batch.png)

Generally we set a fixed batch size to divide the continuous data into several batches (batches).

```python
train_dataset = train_dataset.batch(batch_size=32)
```

The batched data is increased by one dimension, and the size is `batch_size`.

```python
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape, image.dtype)
```

```text
(32, 28, 28, 1) Float32
```

## Customizing Dataset

`mindspore.dataset` provides the loading APIs for some common datasets and standard format datasets. For those datasets that MindSpore does not support yet, it is suggested to load data by constructing customized classes or customized generators. `GeneratorDataset` can help to load dataset based on the logic inside these classes/functions.

`GeneratorDataset` supports constructing customized datasets from random-accessible objects, iterable objects and Python generator, which are explained in detail below.

### Random-accessible Dataset

A Random-accessible dataset is one that implements the `__getitem__` and `__len__` methods, which represents a map from indices/keys to data samples.

For example, when access a dataset with `dataset[idx]`, it should read the idx-th data inside the dataset content.

```python
# Random-accessible object as input source
class RandomAccessDataset:
    def __init__(self):
        self._data = np.ones((5, 2))
        self._label = np.zeros((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)
```

```python
loader = RandomAccessDataset()
dataset = GeneratorDataset(source=loader, column_names=["data", "label"])

for data in dataset:
    print(data)
```

```text
[Tensor(shape=[2], dtype=Float64, value= [ 1.00000000e+00,  1.00000000e+00]), Tensor(shape=[1], dtype=Float64, value= [ 0.00000000e+00])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.00000000e+00,  1.00000000e+00]), Tensor(shape=[1], dtype=Float64, value= [ 0.00000000e+00])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.00000000e+00,  1.00000000e+00]), Tensor(shape=[1], dtype=Float64, value= [ 0.00000000e+00])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.00000000e+00,  1.00000000e+00]), Tensor(shape=[1], dtype=Float64, value= [ 0.00000000e+00])]
[Tensor(shape=[2], dtype=Float64, value= [ 1.00000000e+00,  1.00000000e+00]), Tensor(shape=[1], dtype=Float64, value= [ 0.00000000e+00])]
```

```python
# list, tuple are also supported.
loader = [np.array(0), np.array(1), np.array(2)]
dataset = GeneratorDataset(source=loader, column_names=["data"])

for data in dataset:
    print(data)
```

```text
[Tensor(shape=[], dtype=Int64, value= 2)]
[Tensor(shape=[], dtype=Int64, value= 0)]
[Tensor(shape=[], dtype=Int64, value= 1)]
```

### Iterable Dataset

An iterable dataset is one that implements the `__iter__` and `__next__` methods, which represents an iterator to return data samples gradually. This type of datasets is suitable for cases where random access are expensive or forbidden.

For example, when access a dataset with `iter(dataset)`, it should return a stream of data from a database or a remote server.

```python
# Iterator as input source
class IterableDataset():
    def __init__(self, start, end):
        '''init the class object to hold the data'''
        self.start = start
        self.end = end
    def __next__(self):
        '''iter one data and return'''
        return next(self.data)
    def __iter__(self):
        '''reset the iter'''
        self.data = iter(range(self.start, self.end))
        return self
```

```python
loader = IterableDataset(1, 5)
dataset = GeneratorDataset(source=loader, column_names=["data"])

for d in dataset:
    print(d)
```

```text
[Tensor(shape=[], dtype=Int64, value= 1)]
[Tensor(shape=[], dtype=Int64, value= 2)]
[Tensor(shape=[], dtype=Int64, value= 3)]
[Tensor(shape=[], dtype=Int64, value= 4)]
```

### Generator

Generator also belongs to iterable dataset types, and it can be a Python's generator to return data until the generator throws a `StopIteration` exception.

Example constructs a generator and loads it into the 'GeneratorDataset'.

```python
# Generator
def my_generator(start, end):
    for i in range(start, end):
        yield i
```

```python
# since a generator instance can be only itered once, we need to wrapper it by lambda to generate multiple instances
dataset = GeneratorDataset(source=lambda: my_generator(3, 6), column_names=["data"])

for d in dataset:
    print(d)
```

```text
[Tensor(shape=[], dtype=Int64, value= 3)]
[Tensor(shape=[], dtype=Int64, value= 4)]
[Tensor(shape=[], dtype=Int64, value= 5)]
```
