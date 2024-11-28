# Functional and Cell

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_en/model_train/model_building/functional_and_cell.md)

## Operator Functional Interface

The MindSpore framework provides a rich set of Functional interfaces, defined under `mindspore.ops`, which define operations or calculations directly as functions, without the need to explicitly create an instance of the operator class. The Functional interface provides interfaces including neural network layer functions, mathematical operations, Tensor operations, Parameter operations, differentiation functions, debugging functions, and other types of interfaces, which can be used directly in the `construct` method of `Cell`, or as standalone operations in data processing or model training.

The flow of MindSpore using the Functional interface in `Cell` is shown below:

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def construct(self, x, y):
        output = ops.add(x, y)
        return output

net = MyCell()
x = ms.Tensor([1, 2, 3], ms.float32)
y = ms.Tensor([4, 5, 6], ms.float32)
output = net(x, y)
print(output)
```

The result is as follows:

```text
[5. 7. 9.]
```

The flow of MindSpore standalone use of the Functional interface is shown below:

```python
import mindspore as ms
import mindspore.ops as ops

x = ms.Tensor([1, 2, 3], ms.float32)
y = ms.Tensor([4, 5, 6], ms.float32)
output = ops.add(x, y)
print(output)
```

The result is as follows:

```text
[5. 7. 9.]
```

## Basic Network Elements Cell

The core element of the MindSpore framework, `mindspore.nn.Cell`, is the basic module for building neural networks and is responsible for defining the computational logic of the network. `Cell` not only supports dynamic graphs (PyNative mode) as a base component of a network, but can also be compiled for efficient computational graph execution in static graphs (GRAPH mode). `Cell` defines the computational process of forward propagation through its `construct` method, and can be extended through inheritance mechanisms to implement customized network layers or complex structures. Through the `set_train` method, `Cell` is able to flexibly switch between training and inference modes to accommodate the behavior difference of different operators in the two modes. In addition, `Cell` provides rich APIs, such as mixed accuracy, parameter management, gradient setting, Hook function, recomputation, to support model optimization and training.

The basic `Cell` build process for MindSpore is shown below:

```python
import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def __init__(self, forward_net):
        super(MyCell, self).__init__(auto_prefix=True)
        self.net = forward_net
        self.relu = ops.ReLU()

    def construct(self, x):
        y = self.net(x)
        return self.relu(y)

inner_net = nn.Conv2d(120, 240, 4, has_bias=False)
my_net = MyCell(inner_net)
print(my_net.trainable_params())
```

The result is as follows:

```text
[Parameter (name=net.weight, shape=(240, 120, 4, 4), dtype=Float32, requires_grad=True)]
```

In MindSpore, the name of the parameter is generally composed based on the name of the object defined in `__init__` and the name used in the definition of the parameter, e.g., in the example above, the parameter for the convolution is `net.weight`, where `net` is the name of the object in `self.net = forward_net`, and `weight` is the `name` in Conv2d when defining the parameter for the convolution: `self.weight = Parameter(initializer(self.weight_init, shape), name='weight')`.

### Parameter Management

MindSpore has two kinds of data objects: `Tensor` and `Parameter`. `Tensor` object is only involved in the computation and does not need to perform gradient derivation and parameter update, while the `Parameter` is passed into the optimizer based on its attribute `requires_grad`.

#### Parameter Obtaining

`mindspore.nn.Cell` uses the `parameters_dict`, `get_parameters`, and `trainable_params` interfaces to get the `Parameter` in `Cell`.

- parameters_dict: Get all Parameters in the network structure, returning `OrderedDict` with key as the Parameter name and value as the Parameter value.

- get_parameters: Get all Parameters in the network structure, returning an iterator over `Parameter` in `Cell`.

- trainable_params: Get the properties of `Parameter` where `requires_grad` is `True`, returning a list of trainable Parameters.

When defining the optimizer, use `net.trainable_params()` to get the list of Parameters that need to be updated.

```python
import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
print(net.trainable_params())

for param in net.trainable_params():
    param_name = param.name
    if "bias" in param_name:
        param.requires_grad = False
print(net.trainable_params())
```

The result is as follows:

```text
[Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)]
[Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True)]
```

#### Parameter Saving and Loading

MindSpore provides `load_checkpoint` and `save_checkpoint` methods for Parameter saving and loading. It should be noted that when Parameter is saved, the Parameter list is saved, and when Parameter is loaded, the object must be a Cell.
When Parameter is loaded, it is possible that the Parameter name is not correct and some modification is needed, you can directly construct a new Parameter list to `load_checkpoint` to load into Cell.

```python
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())

ms.save_checkpoint(net, "dense.ckpt")
dense_params = ms.load_checkpoint("dense.ckpt")
print(dense_params)
new_params = {}
for param_name in dense_params:
    print(param_name, dense_params[param_name].data.asnumpy())
    new_params[param_name] = ms.Parameter(ops.ones_like(dense_params[param_name].data), name=param_name)

ms.load_param_into_net(net, new_params)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())
```

The result is as follows:

```text
weight [[-0.0042482  -0.00427286]]
bias [0.]
{'weight': Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), 'bias': Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)}
weight [[-0.0042482  -0.00427286]]
bias [0.]
weight [[1. 1.]]
bias [1.]
```

### Submodule Management

Other Cell instances may be defined as submodules in `mindspore.nn.Cell`. These submodules are integral parts of the network and may themselves contain learnable Parameters (e.g., weights and biases for convolutional layers) and other submodules. This hierarchical module structure allows users to build complex and reusable neural network architectures.

`mindspore.nn.Cell` provides interfaces such as `cells_and_names`, `insert_child_to_cell`, and so on to realize submodule management functions.

```python
from mindspore import nn

class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Add a submodule using insert_child_to_cell
        self.insert_child_to_cell('conv3', nn.Conv2d(64, 128, 3, 1))

        self.sequential_block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sequential_block(x)
        return x

module = MyCell()

# Iterate over all submodules (both direct and indirect) using cells_and_names
for name, cell_instance in module.cells_and_names():
    print(f"Cell name: {name}, type: {type(cell_instance)}")
```

The result is as follows:

```text
Cell name: , type: <class '__main__.MyCell'>
Cell name: conv1, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: conv2, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: conv3, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: sequential_block, type: <class 'mindspore.nn.layer.container.SequentialCell'>
Cell name: sequential_block.0, type: <class 'mindspore.nn.layer.activation.ReLU'>
Cell name: sequential_block.1, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: sequential_block.2, type: <class 'mindspore.nn.layer.activation.ReLU'>
```

### Hook Function

Debugging deep learning network is a task that every practitioner in the field of deep learning needs to face and invest a lot of effort. Since the deep learning network hides the input and output data and the inverse gradient of the intermediate layer operator and only provides the gradient of the network input data (feature quantity, weight), it leads to the inability to accurately perceive the data changes of the intermediate layer operator, which reduces the debugging efficiency. In order to facilitate users to debug the deep learning network accurately and quickly, MindSpore designs the Hook function in the dynamic graph mode. Using hook function can capture the input and output data of the middle layer operator and the inverse gradient.

Currently, `MindSpore.nn.Cell` provides four forms of Hook function in dynamic graph mode, namely: `register_forward_pre_hook`, `register_forward_hook`, `register_backward_hook` and `register_backward_pre_hook` functions. See [Hook Programming](https://www.mindspore.cn/docs/en/r2.4.1/model_train/custom_program/hook_program.html) for details.

### Recomputation

MindSpore uses automatic differentiation in reverse mode to automatically derive the reverse graph based on the forward graph computation flow, and the forward and reverse graphs together form a complete computation graphs. When computing some inverse operators, the results of some forward operators need to be used, resulting in the need for the results of these forward operators to reside in memory until the inverse operators that depend on them have been computed, and the memory occupied by the results of these forward operators will not be reused. This phenomenon pushes up the memory spikes for training, and is particularly significant in large-scale network models.

To solve this problem, the `mindspore.nn.Cell.recompute` interface provides a recomputation function. The recomputation function allows the results of the forward operator to not be saved, allowing the memory to be reused, and then recomputing the forward operator if the forward result is needed when computing the reverse operator. See [recomputation](https://www.mindspore.cn/docs/en/r2.4.1/model_train/parallel/recompute.html) for details.