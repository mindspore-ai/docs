# Data Iteration

[![下载样例代码](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/tutorials/zh_cn/advanced/dataset/mindspore_iterator.ipynb "下载Notebook")  [![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_download_code.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/tutorials/zh_cn/advanced/dataset/mindspore_iterator.py "查看源文件")  [![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/dataset/iterator.ipynb)

An original dataset is read to memory through a dataset loading API, and then transformed through data augmentation to obtain a dataset object. There are two common data iteration methods for the dataset object:

1. Create an `iterator` to iterate data.
2. Transfer data to the Model APIs (such as `model.train` and `model.eval`) of a network model for iteration and training or inference.

## Creating an Iterator

A dataset object can usually create two iterators to traverse data:

1. Tuple iterator. The `create_tuple_iterator` API used for creating a tuple iterator is usually used in `Model.train` internally. The iterated data can be directly used for training.
2. Dictionary iterator. The `create_dict_iterator` API is used for creating a dictionary iterator. In customized `train` mode, you can perform further data processing based on `key` in the dictionary and then transfer the data to the network.

The following uses examples to describe how to use the two iterators:

```python
import mindspore.dataset as ds

# Dataset
np_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

# Load data.
dataset = ds.NumpySlicesDataset(np_data, column_names=["data"], shuffle=False)
```

Then use `create_tuple_iterator` or `create_dict_iterator` to create a data iterator.

```python
# Create a tuple iterator.
print("\n create tuple iterator")
for item in dataset.create_tuple_iterator():
    print("item:\n", item[0])

# Create a dict iterator.
print("\n create dict iterator")
for item in dataset.create_dict_iterator():
    print("item:\n", item["data"])

# Directly traverse dataset objects (equivalent to creating a tuple iterator).
print("\n iterate dataset object directly")
for item in dataset:
    print("item:\n", item[0])

# Use enumerate to traverse (equivalent to creating a tuple iterator).
print("\n iterate dataset using enumerate")
for index, item in enumerate(dataset):
    print("index: {}, item:\n {}".format(index, item[0]))
```

```text
     create tuple iterator
    item:
     [[1 2]
     [3 4]]
    item:
     [[5 6]
     [7 8]]

     create dict iterator
    item:
     [[1 2]
     [3 4]]
    item:
     [[5 6]
     [7 8]]

     iterate dataset object directly
    item:
     [[1 2]
     [3 4]]
    item:
     [[5 6]
     [7 8]]

     iterate dataset using enumerate
    index: 0, item:
     [[1 2]
     [3 4]]
    index: 1, item:
     [[5 6]
     [7 8]]
```

If multiple epochs need to be generated, you can adjust the value of the input parameter `num_epochs` accordingly. Compared with calling the iterator API  for multiple times, directly setting the number of epochs can improve the data iteration performance.

```python
epoch = 2  # Create a tuple iterator to generate data for two epochs.

iterator = dataset.create_tuple_iterator(num_epochs=epoch)

for i in range(epoch):
    print("epoch: ", i)
    for item in iterator:
        print("item:\n", item[0])
```

```text
    epoch:  0
    item:
     [[1 2]
     [3 4]]
    item:
     [[5 6]
     [7 8]]
    epoch:  1
    item:
     [[1 2]
     [3 4]]
    item:
     [[5 6]
     [7 8]]
```

`mindspore.Tensor` is the default output data type of an iterator. If you want to obtain data of the `numpy.ndarray` type, you can set the input parameter `output_numpy=True`.

```python
# The default output type is mindspore.Tensor.
for item in dataset.create_tuple_iterator():
    print("dtype: ", type(item[0]), "\nitem:\n", item[0])

# Set the output type to numpy.ndarray.
for item in dataset.create_tuple_iterator(output_numpy=True):
    print("dtype: ", type(item[0]), "\nitem:\n", item[0])
```

```text
    dtype:  <class 'mindspore.common.tensor.Tensor'>
    item:
     [[1 2]
     [3 4]]
    dtype:  <class 'mindspore.common.tensor.Tensor'>
    item:
     [[5 6]
     [7 8]]
    dtype:  <class 'numpy.ndarray'>
    item:
     [[1 2]
     [3 4]]
    dtype:  <class 'numpy.ndarray'>
    item:
     [[5 6]
     [7 8]]
```

## Using Iterators During Network Training

The following describes how to use the data iterator during network training in a scenario where a linear function is fitted. The expression of the linear function is as follows:

$$output = {x\_0}\\times1 + {x\_1}\\times2 + {x\_2}\\times3 + ··· + {x\_7}\\times8$$

The function is defined as follows:

```python
def func(x):
    """Define a linear function expression."""
    result = []
    for sample in x:
        total = 0
        for i, e in enumerate(sample):
            total += (i+1) * e
        result.append(total)
    return result
```

Use the preceding linear function to customize a training dataset and a validation dataset. When customizing a training dataset, note that the preceding linear function expression has eight unknown numbers. Eight linear independent equations are obtained by substituting data of the training dataset into the preceding linear function, and a value of the unknown number can be obtained by solving the equation.

```python
import numpy as np

class MyTrainData:
    """Customize a training dataset class."""
    def __init__(self):
        """Initialize operations."""
        self.__data = np.array([[[1, 1, 1, 1, 1, 1, 1, 1]],
                                [[1, 1, 1, 1, 1, 1, 1, 0]],
                                [[1, 1, 1, 1, 1, 1, 0, 0]],
                                [[1, 1, 1, 1, 1, 0, 0, 0]],
                                [[1, 1, 1, 1, 0, 0, 0, 0]],
                                [[1, 1, 1, 0, 0, 0, 0, 0]],
                                [[1, 1, 0, 0, 0, 0, 0, 0]],
                                [[1, 0, 0, 0, 0, 0, 0, 0]]]).astype(np.float32)
        self.__label = np.array([func(x) for x in self.__data]).astype(np.float32)

    def __getitem__(self, index):
        """Define a random access function."""
        return self.__data[index], self.__label[index]

    def __len__(self):
        """Define a function for obtaining the dataset size."""
        return len(self.__data)

class MyEvalData:
    """Customize a validation dataset class."""
    def __init__(self):
        """Initialize operations."""
        self.__data = np.array([[[1, 2, 3, 4, 5, 6, 7, 8]],
                                [[1, 1, 1, 1, 1, 1, 1, 1]],
                                [[8, 7, 6, 5, 4, 3, 2, 1]]]).astype(np.float32)
        self.__label = np.array([func(x) for x in self.__data]).astype(np.float32)

    def __getitem__(self, index):
        """Define a random access function."""
        return self.__data[index], self.__label[index]

    def __len__(self):
        """Define a function for obtaining the dataset size."""
        return len(self.__data)
```

The following uses [mindspore.nn.Dense](https://mindspore.cn/docs/en-US/master/api_python/nn/mindspore.nn.Dense.html#mindspore.nn.Dense) to create a customized network. The input of the network is an 8 x 1 matrix.

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class MyNet(nn.Cell):
    """Customize a network."""
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Dense(8, 1, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x
```

Customize network training. The code is as follows:

```python
import mindspore.dataset as ds
import mindspore as ms
from mindspore import amp

def train(dataset, net, optimizer, loss, epoch):
    """Customize a training process."""
    print("--------- Train ---------")

    train_network = amp.build_train_network(net, optimizer, loss)
    for i in range(epoch):
        # Use the data iterator to obtain data.
        for item in dataset.create_dict_iterator():
            data = item["data"]
            label = item["label"]
            loss = train_network(data, label)

        # Print every five epochs.
        if i % 5 == 0:
            print("epoch:{}, loss: {}".format(i, loss))

dataset = ds.GeneratorDataset(MyTrainData(), ["data", "label"], shuffle=True)  # Define the dataset.

epoch = 40                                                  # Define the epoch.
net = MyNet()                                               # Define the network.
loss = nn.MSELoss(reduction="mean")                         # Define the loss function.
optimizer = nn.Momentum(net.trainable_params(), 0.01, 0.9)  # Define the optimizer.

# Start training.
train(dataset, net, optimizer, loss, epoch)
```

```text
    --------- Train ---------
    epoch:0, loss: 117.58063
    epoch:5, loss: 0.28427964
    epoch:10, loss: 0.02881975
    epoch:15, loss: 0.050988887
    epoch:20, loss: 0.0087212445
    epoch:25, loss: 0.040158965
    epoch:30, loss: 0.010140566
    epoch:35, loss: 0.00040051914
```

According to the preceding result, as the number of training times increases, the loss value tends to converge. Then, use the trained network for inference and print the predicted value and target value.

```python
def eval(net, data):
    """Customize the inference process."""
    print("--------- Eval ---------")

    for item in data:
        predict = net(ms.Tensor(item[0]))[0]
        print("predict: {:7.3f}, label: {:7.3f}".format(predict.asnumpy()[0], item[1][0]))

# Start inference.
eval(net, MyEvalData())
```

```text
    --------- Eval ---------
    predict: 203.996, label: 204.000
    predict:  36.012, label:  36.000
    predict: 116.539, label: 120.000
```

The preceding information indicates that the inference result is accurate.

> For details about how to use the data iterator, see the API documents of [create\_tuple\_iterator](https://www.mindspore.cn/docs/en-US/master/api_python/dataset/mindspore.dataset.NumpySlicesDataset.html#mindspore.dataset.NumpySlicesDataset.create_tuple_iterator) and [create\_dict\_iterator](https://www.mindspore.cn/docs/en-US/master/api_python/dataset/mindspore.dataset.NumpySlicesDataset.html#mindspore.dataset.NumpySlicesDataset.create_dict_iterator).

## Data Iteration and Training

After a dataset object is created, you can transfer the dataset object to the `Model` API. The API performs data iteration and sends the dataset object to the network for training or inference. A code example is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import ms_function
from mindspore import nn
import mindspore.dataset as ds
import mindspore.ops as ops

def create_dataset():
    """Create a customized dataset."""
    np_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    np_data = np.array(np_data, dtype=np.float16)
    dataset = ds.NumpySlicesDataset(np_data, column_names=["data"], shuffle=False)
    return dataset

class Net(nn.Cell):
    """Create a neural network."""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()
        self.print = ops.Print()

    @ms_function
    def construct(self, x):
        self.print(x)
        return self.relu(x)

dataset = create_dataset()

network = Net()
model = ms.Model(network)

# The dataset is transferred to the model, and the train API performs data iteration processing.
model.train(epoch=1, train_dataset=dataset)
```
