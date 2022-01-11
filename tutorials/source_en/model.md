# Building a Neural Network

`Ascend` `GPU` `CPU` `Beginner` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/tutorials/source_en/model.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

A neural network model consists of multiple data operation layers. `mindspore.nn` provides various basic network modules.

The following uses LeNet as an example to describe how MindSpore builds a neural network model.

Import the required modules and APIs:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
```

## Defining a Model Class

The `Cell` class of MindSpore is the base class for building all networks and the basic unit of a network. When a neural network is required, you need to inherit the `Cell` class and overwrite the `__init__` and `construct` methods.

```python
class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Define the required operation.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use the defined operation to build a forward network.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

## Model Layers

The following describes the key member functions of the `Cell` class used in LeNet, and then describes how to use the `Cell` class to access model parameters through the instantiation network.

### nn.Conv2d

Add the `nn.Conv2d` layer and add a convolution function to the network to help the neural network extract features.

```python
conv2d = nn.Conv2d(1, 6, 5, has_bias=False, weight_init='normal', pad_mode='valid')
input_x = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)

print(conv2d(input_x).shape)
```

```text
    (1, 6, 28, 28)
```

### nn.ReLU

Add the `nn.ReLU` layer and add a non-linear activation function to the network to help the neural network learn various complex features.

```python
relu = nn.ReLU()
input_x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
output = relu(input_x)

print(output)
```

```text
    [0. 2. 0. 2. 0.]
```

### nn.MaxPool2d

Initialize the `nn.MaxPool2d` layer and down-sample the 6 x 28 x 28 array to a 6 x 14 x 14 array.

```python
max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
input_x = Tensor(np.ones([1, 6, 28, 28]), mindspore.float32)

print(max_pool2d(input_x).shape)
```

```text
    (1, 6, 14, 14)
```

### nn.Flatten

Initialize the `nn.Flatten` layer and convert the 16 x 5 x 5 array into 400 consecutive arrays.

```python
flatten = nn.Flatten()
input_x = Tensor(np.ones([1, 16, 5, 5]), mindspore.float32)
output = flatten(input_x)

print(output.shape)
```

```text
    (1, 400)
```

### nn.Dense

Initialize the `nn.Dense` layer and perform linear transformation on the input matrix.

```python
dense = nn.Dense(400, 120, weight_init='normal')
input_x = Tensor(np.ones([1, 400]), mindspore.float32)
output = dense(input_x)

print(output.shape)
```

```text
    (1, 120)
```

## Model Parameters

The convolutional layer and fully-connected layer in the network will have weights and offsets after being instantiated, and these weight and offset parameters are optimized in subsequent training. In `nn.Cell`, the `parameters_and_names()` method is used to access all parameters.

In the example, we traverse each parameter and display the name and attribute of each layer in the network.

```python
model = LeNet5()
for m in model.parameters_and_names():
    print(m)
```

```text
('conv1.weight', Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)),
('conv2.weight', Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)),
('fc1.weight', Parameter (name=fc1.weight, shape=(120, 400), dtype=Float32, requires_grad=True)),
('fc1.bias', Parameter (name=fc1.bias, shape=(120,), dtype=Float32, requires_grad=True)),
('fc2.weight', Parameter (name=fc2.weight, shape=(84, 120), dtype=Float32, requires_grad=True)),
('fc2.bias', Parameter (name=fc2.bias, shape=(84,), dtype=Float32, requires_grad=True)),
('fc3.weight', Parameter (name=fc3.weight, shape=(10, 84), dtype=Float32, requires_grad=True)),
('fc3.bias', Parameter (name=fc3.bias, shape=(10,), dtype=Float32, requires_grad=True))
```
