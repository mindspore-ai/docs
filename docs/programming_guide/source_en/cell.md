# Cell Building and Its Subclasses

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_en/cell.md)

## Overview

The `Cell` class of MindSpore is the base class for building all networks and the basic unit of a network. When you need to customize a network, you need to inherit the `Cell` class and override the `__init__` and `construct` methods.

Loss functions, optimizers, and model layers are parts of the network structure and can be implemented only by inheriting the `Cell` class. You can also customize them based on service requirements.

The following describes the key member functions of the `Cell` class, the built-in loss functions, optimizers, and model layers of MindSpore implemented based on the `Cell` class, and how to use them, as well as describes how to use the `Cell` class to build a customized network.

## Key Member Functions

### construct

The `Cell` class overrides the `__call__` method. When the `Cell` class instance is called, the `construct` method is executed. The network structure is defined in the `construct` method.

In the following example, a simple network is built to implement the convolution computing function. The operators in the network are defined in `__init__` and used in the `construct` method. The network structure of the case is as follows: `Conv2d` -> `BiasAdd`.

In the `construct` method, `x` is the input data, and `output` is the result obtained after the network structure computation.

```python
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = P.Conv2D(out_channels, kernel_size)
        self.bias_add = P.BiasAdd()
        self.weight = Parameter(initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]))

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```

### parameters_dict

The `parameters_dict` method is used to identify all parameters in the network structure and return `OrderedDict` with key as the parameter name and value as the parameter value.

There are many other methods for returning parameters in the `Cell` class, such as `get_parameters` and `trainable_params`. For details, see [MindSpore API](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/nn/mindspore.nn.Cell.html).

A code example is as follows:

```python
net = Net()
result = net.parameters_dict()
print(result.keys())
print(result['weight'])
```

In the example, `Net` uses the preceding network building case to print names of all parameters on the network and the result of the `weight` parameter.

The following information is displayed:

```text
odict_keys(['weight'])
Parameter (name=weight)
```

### cells_and_names

The `cells_and_names` method is an iterator that returns the name and content of each `Cell` on the network.

The case simply implements the function of obtaining and printing the name of each `Cell`. According to the network structure, there is a `Cell` whose name is `nn.Conv2d`.

`nn.Conv2d` is a convolutional layer encapsulated by MindSpore using `Cell` as the base class. For details, see "Model Layers".

A code example is as follows:

```python
import mindspore.nn as nn

class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')

    def construct(self, x):
        out = self.conv(x)
        return out

net = Net1()
names = []
for m in net.cells_and_names():
    print(m)
    names.append(m[0]) if m[0] else None
print('-------names-------')
print(names)
```

The following information is displayed:

```text
('', Net1<
  (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros, format=NCHW>
  >)
('conv', Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros, format=NCHW>)
-------names-------
['conv']
```

### set_grad

The `set_grad` API is used to construct a backward network. If no parameter is transferred for calling the API, the default value of `requires_grad` is True. This API needs to be used in the scenario where the backward network is computed.

Take `TrainOneStepCell` as an example. Its API function is to perform single-step training on the network. The backward network needs to be computed. Therefore, `set_grad` needs to be used in the initialization method.

A part of the `TrainOneStepCell` code is as follows:

```python
class TrainOneStepCell(Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        ......
```

If using similar APIs such as `TrainOneStepCell`, you do not need to use `set_grad`. The internal encapsulation is implemented.

If you need to customize APIs of this training function, call APIs internally or set `network.set_grad` externally.

## Relationship Between the nn Module and the ops Module

The nn module of MindSpore is a model component implemented by Python. It encapsulates low-level APIs, including various model layers, loss functions, and optimizers.

In addition, nn provides some APIs with the same name as the `Primitive` operator to further encapsulate the `Primitive` operator and provide more friendly APIs.

Reanalyze the case of the `construct` method described above. This case is the simplified content of the `nn.Conv2d` source code of MindSpore, and `P.Conv2D` is internally called. The `nn.Conv2d` convolution API adds the input parameter validation function and determines whether `bias` is used. It is an advanced encapsulated model layer.

```python
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = P.Conv2D(out_channels, kernel_size)
        self.bias_add = P.BiasAdd()
        self.weight = Parameter(initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]))

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```

## Model Layers

MindSpore can use `Cell` as the base class to build the network structure.

To facilitate user operations, MindSpore provides a large number of built-in model layers, which can be directly called by using APIs.

You can also customize a model. For details, see "Building a Customized Network."

### Built-in Model Layers

The MindSpore framework provides abundant APIs at the layer of `mindspore.nn`. The APIs are as follows:

- Activation layer

  The activation layer has a large number of built-in activation functions, which are often used in defining the network structure. The activation function adds a nonlinear operation to the network, so that the network can have a better fitting effect.

  Main APIs include `Softmax`, `Relu`, `Elu`, `Tanh` and `Sigmoid`.

- Basic layer

  The basic layer implements some common basic structures on the network, such as the full connection layer, Onehot encoding, Dropout, and flat layer.

  Main APIs include `Dense`, `Flatten`, `Dropout`, `Norm` and `OneHot`.

- Container layer

  The main function of the container layer is to implement the data structures for storing multiple cells.

  Main APIs include `SequentialCell` and `CellList`.

- Convolutional layer

  Convolutional layer provides some convolution computation functions, such as common convolution, deep convolution, and convolution transposition.

  Main APIs include `Conv2d`, `Conv1d`, `Conv2dTranspose` and `Conv1dTranspose`.

- Pooling layer

  The pooling layer provides computation functions such as average pooling and maximum pooling.

  The main APIs are `AvgPool2d`, `MaxPool2d`, and `AvgPool1d`.

- Embedding layer

  The embedding layer provides the word embedding computation function to map input words into dense vectors.

  The main APIs include `Embedding`, `EmbeddingLookup` and `EmbeddingLookUpSplitMode`.

- Long short-term memory recurrent layer

  The long short-term memory recurrent layer provides the LSTM computation function. `LSTM` internally calls the `LSTMCell` API. The `LSTMCell` is an LSTM unit that performs operations on an LSTM layer. When operations at multiple LSTM network layers are involved, the `LSTM` API is used.

  The main APIs include `LSTM` and `LSTMCell`.

- Normalization layer

  The normalization layer provides some normalization methods, that is, converting data into a mean value and a standard deviation by means of linear transformation or the like.

  Main APIs include `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm` and `GlobalBatchNorm`.

- Mathematical computation layer

The mathematical computation layer provides some computation functions formed by operators, for example, data generation and some other mathematical computations.

  Main APIs include `ReduceLogSumExp`, `Range`, `LinSpace` and `LGamma`.

- Image layer

  The image computation layer provides some functions related to matrix computing to transform and compute image data.

  Main APIs include `ImageGradients`, `SSIM`, `MSSSIM`, `PSNR` and `CentralCrop`.

- Quantization layer

  Quantization is to convert data from the float type to the int type within a data range. Therefore, the quantization layer provides some data quantization methods and model layer structure encapsulation.

  Main APIs include `Conv2dBnAct`, `DenseBnAct`, `Conv2dBnFoldQuant` and `LeakyReLUQuant`.

### Application Cases

Model layers of MindSpore are under `mindspore.nn`. The usage method is as follows:

```python
import mindspore.nn as nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out
```

The preceding network building case shows that the program calls the APIs of the `Conv2d`, `BatchNorm2d`, `ReLU`, `Flatten`, and `Dense` model layers.

It is defined in the `Net` initialization method and runs in the `construct` method. These model layer APIs are connected in sequence to form an executable network.

## Loss Functions

Currently, MindSpore supports the following loss functions: `L1Loss`, `MSELoss`, `SmoothL1Loss`, `SoftmaxCrossEntropyWithLogits`, `SampledSoftmaxLoss`, `BCELoss`, and `CosineEmbeddingLoss`.

All loss functions of MindSpore are implemented by subclasses of `Cell`. Therefore, customized loss functions are also supported. For details about how to build a loss function, see "Building a Customized Network."

### Built-in Loss Functions

- L1Loss

  Computes the absolute value error of two input data for the regression model. The default value of `reduction` is mean. If the value of `reduction` is sum, the loss accumulation result is returned. If the value of `reduction` is none, the result of each loss is returned.

- MSELoss

  Computes the square error of two input data for the regression model. The `reduction` parameter is the same as the `L1Loss` parameter.

- SmoothL1Loss

  `SmoothL1Loss` is the smooth L1 loss function, which is used for the regression model. The default value of the `beta` threshold is 1.

- SoftmaxCrossEntropyWithLogits

  Cross entropy loss function, which is used to classify models. If the tag data is not encoded in the one-hot mode, set `sparse` to True. The default value of `reduction` is none. The meaning of this parameter is the same as that of `L1Loss`.

- CosineEmbeddingLoss

  `CosineEmbeddingLoss` is used to measure the similarity between two inputs and is used for classification models. The default value of `margin` is 0.0. The `reduction` parameter is the same as the `L1Loss` parameter.

### Application Cases

All loss functions of MindSpore are stored in mindspore.nn. The usage method is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.L1Loss()
input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
print(loss(input_data, target_data))
```

The output is as follows:

```text
1.5
```

In this case, two pieces of tensor data are built. The `nn.L1Loss` API is used to define the loss, `input_data` and `target_data` are transferred to the loss, and the L1Loss computation is performed. The result is 1.5. If loss is set to nn.L1Loss(reduction='sum'), the result is 9.0. If loss is set to nn.L1Loss(reduction='none'), the result is [[1. 0. 2.] [1. 2. 3.]].

## Optimization Algorithms

`mindspore.nn.optim` is a module that implements various optimization algorithms in the MindSpore framework. For details, see [Optimization Algorithms](https://www.mindspore.cn/doc/programming_guide/en/r1.1/optim.html)

## Building a Customized Network

Both the network structure and the model layers (e.g. loss functions and optimizers mentioned above) are essentially a `Cell`. Therefore, they can be customized.

Construct a subclass inherited from `Cell`, define the operator and model layer in the `__init__` method, and build the network structure in the `construct` method.

Take the LeNet network as an example. Structure units such as the convolutional layer, pooling layer, and full connection layer are defined in the `__init__` method, and the defined content is connected together in the `construct` method to form a complete LeNet network structure.

The LeNet network is implemented as follows:

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
