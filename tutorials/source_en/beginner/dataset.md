<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/dataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || **Dataset** || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Infer](https://www.mindspore.cn/tutorials/en/master/beginner/infer.html)

# Dataset

Data is the foundation of deep learning, and high-quality data input is beneficial to the entire deep neural network. MindSpore provides Pipeline-based [Data Engine](https://www.mindspore.cn/docs/zh-CN/master/design/data_engine.html) and achieves efficient data preprocessing through [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) and [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html), where Dataset is the start of Pipeline and is used to load raw data. `mindspore.dataset` provides built-in dataset interfaces for loading text, image, audio, etc., and provides interfaces for loading customized datasets.

In addition, MindSpore's domain development library also provides a large number of preloaded datasets that can be downloaded and used with one click through the API. This tutorial will elaborate on different dataset loading methods, common dataset operations and customized dataset methods respectively.

```python
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset, GeneratorDataset
from mindvision import dataset
import matplotlib.pyplot as plt
```

## Loading a Dataset

We use the **Mnist** dataset as a sample to introduce the loading method by using [Vision](https://gitee.com/mindspore/vision/) development library and the `mindspore.dataset` respectively.

### Loading a Dataset by Using Vision

The [Vision](https://gitee.com/mindspore/vision/) development library provides dataset encapsulation based on the customized dataset `mindspore.dataset.GeneratorDataset`. We use its interface for loading with the following parameters to be configured.

- `path`: the path to save the dataset.
- `split`: the dataset category, `train` or `test`.
- `download`: whether to download automatically.

```python
# Download training data from open datasets
training_data = dataset.Mnist(
    path="dataset",
    split="train",
    download=True
)
```

Get the dataset from the automatically downloaded and loaded `training_data`, you can see that its data type is `GeneratorDataset`.

```python
type(training_data.dataset)
```

```text
mindspore.dataset.engine.datasets_user_defined.GeneratorDataset
```

### Dataset Loading by Using mindspore.dataset

`mindspore.dataset` provides a large number of dataset loading interfaces. Here we still take Mnist as an example and load it directly by using the downloaded dataset file. The interface provided by `mindspore.dataset` **only supports decompressed datafiles**, so let's remove the compressed file first.

```python
import os

train_path = "dataset/train"

os.system(f'rm -f {train_path}/*.gz')
os.system(f'ls {train_path}')
```

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
0
```

After the compressed file is deleted and loaded directly, you can see that its data type is MnistDataset.

```python
train_dataset = MnistDataset(train_path, shuffle=False)
type(train_dataset)
```

```text
mindspore.dataset.engine.datasets_vision.MnistDataset
```

## Iterating a Dataset

After the dataset is loaded, the data is generally acquired in an iterative manner and then fed into the neural network for training. You can use the `create_dict_iterator` interface to create a data iterator to iteratively access data. The default type of data to be accessed is `Tensor`. If `output_numpy=True` is set, the type of data to be accessed is `Numpy`.

```python
def visualize(dataset):
    figure = plt.figure(figsize=(4, 4))
    cols, rows = 3, 3

    for idx, (image, label) in enumerate(dataset.create_tuple_iterator()):
        figure.add_subplot(rows, cols, idx + 1)
        plt.title(int(label))
        plt.axis("off")
        plt.imshow(image.asnumpy().squeeze(), cmap="gray")
        if idx == cols * rows - 1:
            break
    plt.show()
```

```text
visualize(train_dataset)
```

## Common Operations on Datasets

The common operations of dataset use the asynchronous execution of `dataset = dataset.operation()` according to The design concept of Pipeline. The execution of the operation returns a new Dataset, at which time no specific operation is executed, but nodes are added to the Pipeline. The whole Pipeline is executed in parallel when iteration is finally performed.

The following are the common operations of datasets.

### shuffle

Random `shuffle` of datasets can eliminate the problem of uneven distribution caused by data alignment. The datasets provided by `mindspore.dataset` can be configured with `shuffle=True` at loading time, or using the following operation:

```python
train_dataset = train_dataset.shuffle(buffer_size=64)

visualize(train_dataset)
```

### map

The `map` is the key operation of data preprocessing, which can add data transforms to a specified column of the dataset, apply data transforms to each element of the column data, and return a new dataset containing the transformed elements. Here data scaling is performed on the Mnist dataset by dividing the image uniformly by 255 and converting the data type from uint8 to float32.

> For the different types of transforms supported by dataset, see [Data Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html).

```python
image, label = next(train_dataset.create_tuple_iterator())
image.shape, image.dtype
```

```text
((28, 28, 1), mindspore.uint8)
```

```python
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
```

Comparing the data before and after map, you can see the data type change.

```python
image, label = next(train_dataset.create_tuple_iterator())
image.shape, image.dtype
```

```text
((28, 28, 1), mindspore.float32)
```

### batch

Packing the dataset into a fixed size `batch` is a compromise method for model optimization using gradient descent with limited hardware resources, which can ensure the randomness of gradient descent and optimize the computational effort. Generally we set a fixed batch size to divide the continuous data into several batches (batches).

```python
train_dataset = train_dataset.batch(batch_size=32)
```

The batched data is increased by one dimension, and the size is `batch_size`.

```python
image, label = next(train_dataset.create_tuple_iterator())
image.shape, image.dtype
```

```text
((32, 28, 28, 1), mindspore.float32)
```

## Customizing Dataset

`mindspore.dataset` provides the loading interface for some common datasets and standard format datasets. For those datasets that MindSpore does not support directly, you can generate datasets by constructing customized dataset classes or customized dataset generator functions, and then implement customized dataset loading through the `GeneratorDataset` interface.

`GeneratorDataset` supports constructing customized datasets from iterable objects, iterators and generator functions, which are explained in detail below.

### Iterable Objects

Iterable object means that Python can use a for loop to traverse over all elements and we can construct an iterable object by implementing the `__getitem__` method and loading it into the `GeneratorDataset`.

```python
# Iterable object as input source
class Iterable:
    def __init__(self):
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)
```

```python
data = Iterable()
dataset = GeneratorDataset(source=data, column_names=["data", "label"])
```

```python
# list, dict, tuple are also iterable object.
dataset = GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])
```

### Iterator

Objects in Python that have `__iter__` and `__next__` methods built in are called iterators (Iterators). The following constructs a simple iterator and loads it into `GeneratorDataset`.

```python
# Iterator as input source
class Iterator:
    def __init__(self):
        self._index = 0
        self._data = np.random.sample((5, 2))
        self._label = np.random.sample((5, 1))

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)
```

```python
data = Iterator()
dataset = GeneratorDataset(source=data, column_names=["data", "label"])
```
