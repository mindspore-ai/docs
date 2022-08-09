# Building a Network

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/network/forward.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The `Cell` class of MindSpore is the base class for setting up all networks and the basic unit of a network. When customizing a network, you need to inherit the `Cell` class. The following describes the basic network unit `Cell` and customized feedforward network.

The following describes the build of the feedforward network model and the basic units of the network model. Because training is not involved, there is no backward propagation or backward graph.

![learningrate.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/network/images/introduction3.png)

## Network Basic Unit: Cell

In order to customize a network, you need to inherit the `Cell` class and overwrite the `__init__` and `construct` methods. Loss functions, optimizers, and model layers are parts of the network structure and can implement functions by only inheriting the `Cell` class. You can also customize them based on service requirements.

The following describes the key member functions of `Cell`.

### construct

The `Cell` class overrides the `__call__` method. When the `Cell` class instance is called, the `construct` method is executed. The network structure is defined in the `construct` method.

In the following example, a simple convolutional network is built. The convolutional network is defined in `__init__`. Input data `x` is transferred to the `construct` method to perform convolution computation and the computation result is returned.

```python
from mindspore import nn


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(10, 20, 3, has_bias=True, weight_init='normal')

    def construct(self, x):
        out = self.conv(x)
        return out
```

### Obtaining Network Parameters

In `nn.Cell`, methods that return parameters are `parameters_dict`, `get_parameters`, and `trainable_params`.

- `parameters_dict`: obtains all parameters in the network structure and returns an OrderedDict with `key` as the parameter name and `value` as the parameter value.
- `get_parameters`: obtains all parameters in the network structure and returns the iterator of the Parameter in the Cell.
- `trainable_params`: obtains the attributes whose `requires_grad` is True in Parameter and returns the list of trainable parameters.

The following examples use the preceding methods to obtain and print network parameters.

```python
net = Net()

# Obtain all parameters in the network structure.
result = net.parameters_dict()
print("parameters_dict of result:\n", result)

# Obtain all parameters in the network structure.
print("\nget_parameters of result:")
for m in net.get_parameters():
    print(m)

# Obtain the list of trainable parameters.
result = net.trainable_params()
print("\ntrainable_params of result:\n", result)
```

```text
parameters_dict of result:
OrderedDict([('conv.weight', Parameter (name=conv.weight, shape=(20, 10, 3, 3), dtype=Float32, requires_grad=True)), ('conv.bias', Parameter (name=conv.bias, shape=(20,), dtype=Float32, requires_grad=True))])

get_parameters of result:
Parameter (name=conv.weight, shape=(20, 10, 3, 3), dtype=Float32, requires_grad=True)
Parameter (name=conv.bias, shape=(20,), dtype=Float32, requires_grad=True)

trainable_params of result:
[Parameter (name=conv.weight, shape=(20, 10, 3, 3), dtype=Float32, requires_grad=True), Parameter (name=conv.bias, shape=(20,), dtype=Float32, requires_grad=True)]
```

### Related Attributes

1. cells_and_names

    The `cells_and_names` method is an iterator that returns the name and content of each `Cell` on the network. A code example is as follows:

    ```python
    net = Net()
    for m in net.cells_and_names():
        print(m)
    ```

    ```text
        ('', Net<
        (conv): Conv2d<input_channels=10, output_channels=20, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=True, weight_init=normal, bias_init=zeros, format=NCHW>
        >)
        ('conv', Conv2d<input_channels=10, output_channels=20, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=True, weight_init=normal, bias_init=zeros, format=NCHW>)
    ```

2. set_grad

    `set_grad` is used to specify whether the network needs to compute the gradient. If no parameter is transferred, `requires_grad` is set to True by default. When the feedforward network is executed, a backward network for computing gradients is built. The `TrainOneStepCell` and `GradOperation` APIs do not need to use `set_grad` because they have been implemented internally. If you need to customize the APIs of this training function, you need to set `set_grad` internally or externally.

    ```python
    class CustomTrainOneStepCell(nn.Cell):
        def __init__(self, network, optimizer, sens=1.0):
            """There are three input parameters: training network, optimizer, and backward propagation scaling ratio."""
            super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
            self.network = network      # Feedforward network
            self.network.set_grad()     # Build a backward network for computing gradients.
            self.optimizer = optimizer  # Optimizer
    ```

    For details about the `CustomTrainOneStepCell` code, see [Customized Training and Evaluation Networks](https://www.mindspore.cn/tutorials/en/master/advanced/train/train_eval.html).

3. set_train

    The `set_train` API determines whether the model is in training mode. If no parameter is transferred, the `mode` attribute is set to True by default.

    When implementing a network with different training and inference structures, you can use the `training` attribute to distinguish the training scenario from the inference scenario. When `mode` is set to True, the training scenario is used. When `mode` is set to False, the inference scenario is used.

    For the `nn.Dropout` operator in MindSpore, two execution logics are distinguished based on the `mode` attribute of the `Cell`. If the value of `mode` is False, the input is directly returned. If the value of `mode` is True, the operator is executed.

    ```python
    import numpy as np
    import mindspore as ms

    x = ms.Tensor(np.ones([2, 2, 3]), ms.float32)
    net = nn.Dropout(keep_prob=0.7)

    # Start training.
    net.set_train()
    output = net(x)
    print("training result:\n", output)

    # Start inference.
    net.set_train(mode=False)
    output = net(x)
    print("\ninfer result:\n", output)
    ```

    ```text
        training result:
        [[[1.4285715 1.4285715 1.4285715]
        [1.4285715 0.        0.       ]]

        [[1.4285715 1.4285715 1.4285715]
        [1.4285715 1.4285715 1.4285715]]]

        infer result:
        [[[1. 1. 1.]
        [1. 1. 1.]]

        [[1. 1. 1.]
        [1. 1. 1.]]]
    ```

4. to_float

    The `to_float` API recursively configures the forcible conversion types of the current `Cell` and all sub-`Cell`s so that the current network structure uses a specific float type. This API is usually used in mixed precision scenarios.

    The following example uses the float32 and float16 types to compute the `nn.dense` layer and prints the data type of the output result.

    ```python
    import numpy as np
    from mindspore import nn
    import mindspore as ms

    # float32 is used for computation.
    x = ms.Tensor(np.ones([2, 2, 3]), ms.float32)
    net = nn.Dense(3, 2)
    output = net(x)
    print(output.dtype)

    # float16 is used for computation.
    net1 = nn.Dense(3, 2)
    net1.to_float(ms.float16)
    output = net1(x)
    print(output.dtype)
    ```

    ```text
        Float32
        Float16
    ```

## Building a Network

When building a network, you can inherit the `nn.Cell` class, declare the definition of each layer in the `__init__` constructor, and implement the connection relationship between layers in `construct` to complete the build of the feedforward neural network.

The `mindspore.ops` module provides the implementation of basic operators, such as neural network operators, array operators, and mathematical operators.

The `mindspore.nn` module further encapsulates basic operators. You can flexibly use different operators as required.

In addition, to better build and manage complex networks, `mindspore.nn` provides two types of containers to manage submodules or model layers on the network: `nn.CellList` and `nn.SequentialCell`.

### ops-based Network Build

The [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) module provides the implementation of basic operators, such as neural network operators, array operators, and mathematical operators.

You can use operators in `mindspore.ops` to build a simple algorithm $f(x)=x^2+w$. The following is an example:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.weight = ms.Parameter(ms.Tensor(np.array([2, 2, 2]), ms.float32))

    def construct(self, x):
        return self.add(self.mul(x, x), self.weight)

net = Net()
input = ms.Tensor(np.array([1, 2, 3]), ms.float32)
output = net(input)

print(output)
```

```text
    [ 3.  6. 11.]
```

### nn-based Network Build

Although various operators provided by the `mindspore.ops` module can basically meet network build requirements, [mindspore.nn](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#) further encapsulates the `mindspore.ops` operator to provide more convenient and easy-to-use APIs in complex deep networks.

The `mindspore.nn` module mainly includes a convolutional layer (such as `nn.Conv2d`), a pooling layer (such as `nn.MaxPool2d`), and a non-linear activation function (such as `nn.ReLU`), a loss functions (such as `nn.LossBase`) and an optimizer (such as `nn.Momentum`) that are commonly used in a neural network to facilitate user operations.

In the following sample code, the `mindspore.nn` module is used to build a Conv + Batch Normalization + ReLu model network.

```python
import numpy as np
from mindspore import nn

class ConvBNReLU(nn.Cell):
    def __init__(self):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

net = ConvBNReLU()
print(net)
```

```text
    ConvBNReLU<
      (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      >
```

### Container-based Network Build

To facilitate managing and forming a more complex network, `mindspore.nn` provides containers to manage submodel blocks or model layers on the network through either `nn.CellList` and `nn.SequentialCell`.

1. CellList-based Network Build

    A Cell built using `nn.CellList` can be either a model layer or a built network subblock. `nn.CellList` supports the `append`, `extend`, and `insert` methods.

    When running the network, you can use the for loop in the construct method to obtain the output result.

    - `append(cell)`: adds a cell to the end of the list.
    - `extend (cells)`: adds cells to the end of the list.
    - `insert(index, cell)`: inserts a given cell before the given index in the list.

    The following uses `nn.CellList` to build and execute a network that contains a previously defined model subblock ConvBNReLU, a Conv2d layer, a BatchNorm2d layer, and a ReLU layer in sequence:

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore import nn

    class MyNet(nn.Cell):

        def __init__(self):
            super(MyNet, self).__init__()
            layers = [ConvBNReLU()]
            # Use CellList to manage the network.
            self.build_block = nn.CellList(layers)

            # Use the append method to add the Conv2d and ReLU layers.
            self.build_block.append(nn.Conv2d(64, 4, 4))
            self.build_block.append(nn.ReLU())

            # Use the insert method to insert BatchNorm2d between the Conv2d layer and the ReLU layer.
            self.build_block.insert(-1, nn.BatchNorm2d(4))

        def construct(self, x):
            # Use the for loop to execute the network.
            for layer in self.build_block:
                x = layer(x)
            return x

    net = MyNet()
    print(net)
    ```

    ```text
        MyNet<
        (build_block): CellList<
            (0): ConvBNReLU<
            (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
            (relu): ReLU<>
            >
            (1): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (2): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.2.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.2.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.2.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.2.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
            (3): ReLU<>
            >
        >
    ```

    Input data into the network model.

    ```python
    input = ms.Tensor(np.ones([1, 3, 64, 32]), ms.float32)
    output = net(input)
    print(output.shape)
    ```

    ```text
        (1, 4, 64, 32)
    ```

2. SequentialCell-based Network Build

    Use `nn.SequentialCell` to build a Cell sequence container. Submodules can be input in List or OrderedDict format.

    Different from `nn.CellList`, the `nn.SequentialCell` class implements the `construct` method and can directly output results.

    The following example uses `nn.SequentialCell` to build a network. The input is in List format. The network structure contains a previously defined model subblock ConvBNReLU, a Conv2d layer, a BatchNorm2d layer, and a ReLU layer in sequence.

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore import nn

    class MyNet(nn.Cell):

        def __init__(self):
            super(MyNet, self).__init__()

            layers = [ConvBNReLU()]
            layers.extend([nn.Conv2d(64, 4, 4),
                        nn.BatchNorm2d(4),
                        nn.ReLU()])
            self.build_block = nn.SequentialCell(layers) # Use SequentialCell to manage the network.

        def construct(self, x):
            return self.build_block(x)

    net = MyNet()
    print(net)
    ```

    ```text
        MyNet<
        (build_block): SequentialCell<
            (0): ConvBNReLU<
            (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
            (relu): ReLU<>
            >
            (1): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (2): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.2.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.2.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.2.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.2.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
            (3): ReLU<>
            >
        >
    ```

    Input data into the network model.

    ```python
    input = ms.Tensor(np.ones([1, 3, 64, 32]), ms.float32)
    output = net(input)
    print(output.shape)
    ```

    ```text
        (1, 4, 64, 32)
    ```

    The following example uses `nn.SequentialCell` to build a network. The input is in OrderedDict format.

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore import nn
    from collections import OrderedDict

    class MyNet(nn.Cell):

        def __init__(self):
            super(MyNet, self).__init__()
            layers = OrderedDict()

            # Add cells to the dictionary.
            layers["ConvBNReLU"] = ConvBNReLU()
            layers["conv"] = nn.Conv2d(64, 4, 4)
            layers["norm"] = nn.BatchNorm2d(4)
            layers["relu"] = nn.ReLU()

            # Use SequentialCell to manage the network.
            self.build_block = nn.SequentialCell(layers)

        def construct(self, x):
            return self.build_block(x)

    net = MyNet()
    print(net)

    input = ms.Tensor(np.ones([1, 3, 64, 32]), ms.float32)
    output = net(input)
    print(output.shape)
    ```

    ```text
        MyNet<
        (build_block): SequentialCell<
            (ConvBNReLU): ConvBNReLU<
            (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.ConvBNReLU.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.ConvBNReLU.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.ConvBNReLU.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.ConvBNReLU.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
            (relu): ReLU<>
            >
            (conv): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (norm): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.norm.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.norm.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.norm.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.norm.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
            (relu): ReLU<>
            >
        >
        (1, 4, 64, 32)
    ```

## Relationship Between nn and ops

The `mindspore.nn` module is a model component implemented by Python. It encapsulates low-level APIs, including various model layers, loss functions, and optimizers related to neural network models.

In addition, `mindspore.nn` provides some APIs with the same name as the `mindspore.ops` operators to further encapsulate the `mindspore.ops` operators and provide more friendly APIs. You can also use the `mindspore.ops` operators to customize a network based on the actual situation.

The following example uses the `mindspore.ops.Conv2D` operator to implement the convolution computation function, that is, the `nn.Conv2d` operator function.

```python
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.initializer import initializer


class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = ops.Conv2D(out_channels, kernel_size)
        self.bias_add = ops.BiasAdd()
        self.weight = ms.Parameter(
            initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]),
            name='conv.weight')
        self.bias = ms.Parameter(initializer('normal', [out_channels]), name='conv.bias')

    def construct(self, x):
        """Input data x."""
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```
