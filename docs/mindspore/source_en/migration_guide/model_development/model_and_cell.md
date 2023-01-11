# Network Construction

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/model_development/model_and_cell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Before reading this section, read the tutorials [Loss Function](https://www.mindspore.cn/tutorials/en/master/advanced/modules/loss.html) on the MindSpore official website first.

## Network Basic Unit: Cell

MindSpore uses [Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell) to construct graphs. You need to define a class that inherits the `Cell` base class, declare the required APIs and submodules in `init`, and perform calculation in `construct`. `Cell` compiles a computational graph in `GRAPH_MODE` (static graph mode). It is used as the basic module of neural network in `PYNATIVE_MODE` (dynamic graph mode). The basic `Cell` setup process is as follows:

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

```text
    [Parameter (name=net.weight, shape=(240, 120, 4, 4), dtype=Float32, requires_grad=True)]
```

A parameter name is generally formed based on an object name defined by `__init__` and a name used during parameter definition. For example, in the foregoing example, a convolutional parameter name is `net.weight`, where `net` is an object name in `self.net = forward_net`, and `weight` is `name`: `self.weight = Parameter(initializer(self.weight_init, shape), name='weight')` when a convolutional parameter is defined in Conv2d.

To align parameter names, you may not need to add object names. The cell provides the `auto_prefix` interface to determine whether to add object names to parameter names in the cell. The default value is `True`, that is, add object names. If `auto_prefix` is set to `False`, the `name` of `Parameter` in the preceding example is `weight`.

### Unit Test

After the `Cell` is set up, you are advised to build a unit test method for each `Cell` and compare it with the benchmarking code. In the preceding example, the PyTorch build code is as follows:

```python
import torch.nn as torch_nn

class MyCell_pt(torch_nn.Module):
    def __init__(self, forward_net):
        super(MyCell_pt, self).__init__()
        self.net = forward_net
        self.relu = torch_nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        return self.relu(y)

inner_net_pt = torch_nn.Conv2d(120, 240, kernel_size=4, bias=False)
pt_net = MyCell_pt(inner_net_pt)
for i in pt_net.parameters():
    print(i.shape)
```

```text
    torch.Size([240, 120, 4, 4])
```

With the script for building the `Cell`, you need to use the same input data and parameters to compare the output.

```python
import numpy as np
import mindspore as ms
import torch

x = np.random.uniform(-1, 1, (2, 120, 12, 12)).astype(np.float32)
for m in pt_net.modules():
    if isinstance(m, torch_nn.Conv2d):
        torch_nn.init.constant_(m.weight, 0.1)

for _, cell in my_net.cells_and_names():
    if isinstance(cell, nn.Conv2d):
        cell.weight.set_data(ms.common.initializer.initializer(0.1, cell.weight.shape, cell.weight.dtype))

y_ms = my_net(ms.Tensor(x))
y_pt = pt_net(torch.from_numpy(x))
diff = np.max(np.abs(y_ms.asnumpy() - y_pt.detach().numpy()))
print(diff)

# ValueError: operands could not be broadcast together with shapes (2,240,12,12) (2,240,9,9)
```

The output of MindSpore is different from that of PyTorch. Why?

According to the [Function Differences with torch.nn.Conv2d](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/Conv2d.html), the default parameters of `Conv2d` are different in MindSpore and PyTorch.
By default, MindSpore uses the `same` mode, and PyTorch uses the `pad` mode. During migration, you need to modify the `pad_mode` of MindSpore `Conv2d`.

```python
import numpy as np
import mindspore as ms
import torch

inner_net = nn.Conv2d(120, 240, 4, has_bias=False, pad_mode="pad")
my_net = MyCell(inner_net)

# Construct random input.
x = np.random.uniform(-1, 1, (2, 120, 12, 12)).astype(np.float32)
for m in pt_net.modules():
    if isinstance(m, torch_nn.Conv2d):
        # Fixed PyTorch initialization parameter
        torch_nn.init.constant_(m.weight, 0.1)

for _, cell in my_net.cells_and_names():
    if isinstance(cell, nn.Conv2d):
        # Fixed MindSpore initialization parameter
        cell.weight.set_data(ms.common.initializer.initializer(0.1, cell.weight.shape, cell.weight.dtype))

y_ms = my_net(ms.Tensor(x))
y_pt = pt_net(torch.from_numpy(x))
diff = np.max(np.abs(y_ms.asnumpy() - y_pt.detach().numpy()))
print(diff)
```

```text
    2.9355288e-06
```

The overall error is about 0.01%, which basically meets the expectation. **During cell migration, you are advised to perform a unit test on each cell to ensure migration consistency.**

### Common Methods of Cells

`Cell` is the basic unit of the neural network in MindSpore. It provides many flag setting and easy-to-use methods. The following describes some common methods.

#### Manual Mixed-precision

MindSpore provides an auto mixed precision method. For details, see the amp_level attribute in [Model](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model).

However, sometimes the hybrid precision policy is expected to be more flexible during network development. MindSpore also provides the [to_float](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.to_float) method to manually add hybrid precision.

`to_float(dst_type)`: adds type conversion to the input of the `Cell` and all child `Cell` to run with a specific floating-point type.

If `dst_type` is `ms.float16`, all inputs of `Cell` (including input, `Parameter`, and `Tensor` used as constants) will be converted to `float16`. For example, if you want to change all BNs and losses in a network to the `float32` type and other operations to the `float16` type, run the following command:

```python
import mindspore as ms
from mindspore import nn

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            nn.Conv2d(3, 12, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(12, 4, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.pool = nn.AdaptiveMaxPool2d((5, 5))
        self.fc = nn.Dense(100, 10)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view((-1, 100))
        out = nn.Dense(x)
        return out

net = Network()
net.to_float(ms.float16)  #Add the float16 flag to all operations in the net. The framework adds the cast method to the input during compilation.
for _, cell in net.cells_and_names():
    if isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        cell.to_float(ms.float32)

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean').to_float(ms.float32)
net_with_loss = nn.WithLossCell(net, loss_fn=loss)
```

The customized `to_float` conflicts with the `amp_level` in the model. If the customized mixing precision is used, do not set the `amp_level` in the model.

#### Customizing Initialization Parameters

Generally, the high-level API encapsulated by MindSpore initializes parameters by default. Sometimes, the initialization distribution is inconsistent with the required initialization and PyTorch initialization. In this case, you need to customize initialization. [Initializing Network Arguments](https://mindspore.cn/tutorials/en/master/advanced/modules/initializer.html#customized-parameter-initialization) describes a method of initializing parameters by using API attributes. This section describes a method of initializing parameters by using Cell.

For details about the parameters, see [Network Parameters](https://mindspore.cn/tutorials/zh-CN/master/advanced/modules/initializer.html). This section uses `Cell` as an example to describe how to obtain all parameters in `Cell` and how to initialize the parameters in `Cell`.

> Note that the method described in this section cannot be performed in `construct`. To change the value of a parameter on the network, use [assign](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.assign.html).

[set_data(data, slice_shape=False)](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html?highlight=set_data#mindspore.Parameter.set_data) sets parameter data.

For details about the parameter initialization methods supported by MindSpore, see [mindspore.common.initializer](https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html). You can also directly transfer a defined [Parameter](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter) object.

```python
import math
import mindspore as ms
from mindspore import nn

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            nn.Conv2d(3, 12, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(12, 4, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.pool = nn.AdaptiveMaxPool2d((5, 5))
        self.fc = nn.Dense(100, 10)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view((-1, 100))
        out = nn.Dense(x)
        return out

net = Network()

for _, cell in net.cells_and_names():
    if isinstance(cell, nn.Conv2d):
        cell.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
            cell.weight.shape, cell.weight.dtype))
    elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
        cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
        cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
    elif isinstance(cell, (nn.Dense)):
        cell.weight.set_data(ms.common.initializer.initializer(
            ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
            cell.weight.shape, cell.weight.dtype))
        cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))
```

#### Freezing Parameters

`Parameter` has a `requires_grad` attribute to determine whether to update parameters. When `requires_grad=False`, `Parameter` is equivalent to the `buffer` object of PyTorch.

You can obtain the parameter list in `Cell` through `parameters_dict`, `get_parameters`, and `trainable_params` of the cell.

- parameters_dict: obtains all parameters in the network structure and returns an `OrderedDict` with `key` as the parameter name and `value` as the parameter value.

- get_parameters: obtains all parameters in the network structure and returns the iterator of the `Parameter` in the `Cell`.

- trainable_params: obtains the attributes whose `requires_grad` is `True` in `Parameter` and returns the list of trainable parameters.

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

```text
    [Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)]
    [Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True)]
```

When defining an optimizer, use `net.trainable_params()` to obtain the list of parameters that need to be updated.

In addition to setting the parameter `requires_grad=False` not to update the parameter, you can also use `stop_gradient` to block gradient calculation to freeze the parameter. When will `requires_grad=False` and `stop_gradient` be used?

![parameter-freeze](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_en/migration_guide/model_development/images/parameter_freeze.png)

As shown in the preceding figure, the `requires_grad=False` does not update some parameters, but the backward gradient calculation is normal.
The `stop_gradient` directly performs backward gradient. When there is no parameter to be trained before the parameter to be frozen, the two parameters are equivalent in function.
However, `stop_gradient` is faster (with less backward gradient calculations).
If there are parameters to be trained before the frozen parameters, only `requires_grad=False` can be used.
In addition, `stop_gradient` needs to be added to the computational link of the network, acting on the Tensor.

```python
a = A(x)
a = ops.stop_gradient(a)
y = B(a)
```

#### Saving and Loading Parameters

MindSpore provides the `load_checkpoint` and `save_checkpoint` methods for saving and loading parameters. Note that when a parameter is saved, the parameter list is saved. When a parameter is loaded, the object must be a cell.
When loading parameters, you may need to modify the parameter names. In this case, you can directly construct a new parameter list for the `load_checkpoint` to load the parameter list to the cell.

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

```text
    weight [[-0.0042482  -0.00427286]]
    bias [0.]
    {'weight': Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), 'bias': Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)}
    weight [[-0.0042482  -0.00427286]]
    bias [0.]
    weight [[1. 1.]]
    bias [1.]
```

### Dynamic and Static Graphs

For `Cell`, MindSpore provides two image modes: `GRAPH_MODE` (static image) and `PYNATIVE_MODE` (dynamic image). For details, see [Dynamic Image and Static Graphs](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html).

The **inference** behavior of the model in `PyNative` mode is the same as that of common Python code. However, during training, **once a tensor is converted into NumPy for other operations, the gradient of the network is truncated, which is equivalent to detach of PyTorch**.

When `GRAPH_MODE` is used, syntax restrictions usually occur. In this case, graph compilation needs to be performed on the Python code. However, MindSpore does not support the complete Python syntax set. Therefore, there are some restrictions on compiling the `construct` function. For details about the restrictions, see [MindSpore Static Graph Syntax](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

#### Common Restrictions

Compared with the detailed syntax description, the common restrictions are as follows:

- Scenario 1

    Restriction: During image composition (construct functions or functions modified by ms_function), do not invoke other Python libraries, such as NumPy and scipy. Related processing must be moved forward to the `__init__` phase.
    Measure: Use the APIs provided by MindSpore to replace the functions of other Python libraries. The processing of constants can be moved forward to the `__init__` phase.

- Scenario 2

    Restriction: Do not use user-defined types during graph build. Instead, use the data types provided by MindSpore and basic Python types. You can use the tuple/list combination based on these types.
    Measure: Use basic types for combination. You can increase the number of function parameters. There is no restriction on the input parameters of the function, and variable-length input can be used.

- Scenario 3

    Restriction: Do not perform multi-thread or multi-process processing on data during image composition.
    Measure: Avoid multi-thread processing on the network.

### Customized Backward Network Construction

Sometimes, MindSpore does not support some processing and needs to use some third-party library methods. However, we do not want to truncate the network gradient. In this case, what should we do? This section describes how to customize backward network construction to avoid this problem in `PYNATIVE_MODE`.

In this scenario, a value greater than 0.5 needs to be randomly selected, and the shape of each batch is fixed to `max_num`. However, the random put-back operation is not supported by MindSpore APIs. In this case, NumPy is used for computation in `PYNATIVE_MODE`, and then a gradient propagation process is constructed.

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_seed(1)

class MySampler(nn.Cell):
    # Customize a sampler and select `max_num` values greater than 0.5 in each batch.
    def __init__(self, max_num):
        super(MySampler, self).__init__()
        self.max_num = max_num

    def random_positive(self, x):
        # Method of the third-party library NumPy. Select a position greater than 0.5.
        pos = np.where(x > 0.5)[0]
        pos_indice = np.random.choice(pos, self.max_num)
        return pos_indice

    def construct(self, x):
        # Forward Network Construction
        batch = x.shape[0]
        pos_value = []
        pos_indice = []
        for i in range(batch):
            a = x[i].asnumpy()
            pos_ind = self.random_positive(a)
            pos_value.append(ms.Tensor(a[pos_ind], ms.float32))
            pos_indice.append(ms.Tensor(pos_ind, ms.int32))
        pos_values = ops.stack(pos_value, axis=0)
        pos_indices = ops.stack(pos_indice, axis=0)
        print("pos_values forword", pos_values)
        print("pos_indices forword", pos_indices)
        return pos_values, pos_indices

x = ms.Tensor(np.random.uniform(0, 3, (2, 5)), ms.float32)
print("x", x)
sampler = MySampler(3)
pos_values, pos_indices = sampler(x)
grad = ops.GradOperation(get_all=True)(sampler)(x)
print("dx", grad)
```

```text
    x [[1.2510660e+00 2.1609735e+00 3.4312444e-04 9.0699774e-01 4.4026768e-01]
     [2.7701578e-01 5.5878061e-01 1.0366821e+00 1.1903024e+00 1.6164502e+00]]
    pos_values forword [[0.90699774 2.1609735  0.90699774]
     [0.5587806  1.6164502  0.5587806 ]]
    pos_indices forword [[3 1 3]
     [1 4 1]]
    pos_values forword [[0.90699774 1.251066   2.1609735 ]
     [1.1903024  1.1903024  0.5587806 ]]
    pos_indices forword [[3 0 1]
     [3 3 1]]
    dx (Tensor(shape=[2, 5], dtype=Float32, value=
    [[0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
     [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]),)
```

When we do not construct this backward process, the gradient will be truncated because the numpy method is used to calculate the `pos_value`.
As shown in the preceding information, the value of `dx` is all 0s. In addition, you may find that `pos_values forword` and `pos_indices forword` are printed twice in this process. This is because the forward graph is constructed again when the backward graph is constructed in `PYNATIVE_MODE`. As a result, the forward graph is constructed twice and the backward graph is constructed once, which wastes training resources. In some cases, precision problems may occur. For example, in the case of BatchNorm, `moving_mean` and `moving_var` are updated during forward running. As a result, `moving_mean` and `moving_var` are updated twice during one training.
To avoid this scenario, MindSpore has a method `set_grad()` for `Cell`. In `PYNATIVE_MODE` mode, the framework synchronously constructs the backward process when constructing the forward process. In this way, the forward process is not executed when the backward process is executed.

```python
x = ms.Tensor(np.random.uniform(0, 3, (2, 5)), ms.float32)
print("x", x)
sampler = MySampler(3).set_grad()
pos_values, pos_indices = sampler(x)
grad = ops.GradOperation(get_all=True)(sampler)(x)
print("dx", grad)
```

```text
    x [[1.2519144  1.6760695  0.42116082 0.59430444 2.4022336 ]
     [2.9047847  0.9402725  2.076968   2.6291676  2.68382   ]]
    pos_values forword [[1.2519144 1.2519144 1.6760695]
     [2.6291676 2.076968  0.9402725]]
    pos_indices forword [[0 0 1]
     [3 2 1]]
    dx (Tensor(shape=[2, 5], dtype=Float32, value=
    [[0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
     [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]]),)
```

Now, let's see how to [customize backward network construction](https://mindspore.cn/tutorials/zh-CN/master/advanced/modules/layer.html#自定义cell反向).

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_seed(1)

class MySampler(nn.Cell):
    # Customize a sampler and select `max_num` values greater than 0.5 in each batch.
    def __init__(self, max_num):
        super(MySampler, self).__init__()
        self.max_num = max_num

    def random_positive(self, x):
        # Method of the third-party library NumPy. Select a position greater than 0.5.
        pos = np.where(x > 0.5)[0]
        pos_indice = np.random.choice(pos, self.max_num)
        return pos_indice

    def construct(self, x):
        # Forward network construction
        batch = x.shape[0]
        pos_value = []
        pos_indice = []
        for i in range(batch):
            a = x[i].asnumpy()
            pos_ind = self.random_positive(a)
            pos_value.append(ms.Tensor(a[pos_ind], ms.float32))
            pos_indice.append(ms.Tensor(pos_ind, ms.int32))
        pos_values = ops.stack(pos_value, axis=0)
        pos_indices = ops.stack(pos_indice, axis=0)
        print("pos_values forword", pos_values)
        print("pos_indices forword", pos_indices)
        return pos_values, pos_indices

    def bprop(self, x, out, dout):
        # Backward network construction
        pos_indices = out[1]
        print("pos_indices backward", pos_indices)
        grad_x = dout[0]
        print("grad_x backward", grad_x)
        batch = x.shape[0]
        dx = []
        for i in range(batch):
            dx.append(ops.UnsortedSegmentSum()(grad_x[i], pos_indices[i], x.shape[1]))
        return ops.stack(dx, axis=0)

x = ms.Tensor(np.random.uniform(0, 3, (2, 5)), ms.float32)
print("x", x)
sampler = MySampler(3).set_grad()
pos_values, pos_indices = sampler(x)
grad = ops.GradOperation(get_all=True)(sampler)(x)
print("dx", grad)
```

```text
    x [[1.2510660e+00 2.1609735e+00 3.4312444e-04 9.0699774e-01 4.4026768e-01]
     [2.7701578e-01 5.5878061e-01 1.0366821e+00 1.1903024e+00 1.6164502e+00]]
    pos_values forword [[0.90699774 2.1609735  0.90699774]
     [0.5587806  1.6164502  0.5587806 ]]
    pos_indices forword [[3 1 3]
     [1 4 1]]
    pos_indices backward [[3 1 3]
     [1 4 1]]
    grad_x backward [[1. 1. 1.]
     [1. 1. 1.]]
    dx (Tensor(shape=[2, 5], dtype=Float32, value=
    [[0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 2.00000000e+000, 0.00000000e+000],
     [0.00000000e+000, 2.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000]]),)
```

The `bprop` method is added to the `MySampler` class. The input of this method is forward input (expanded write), forward output (a tuple), and output gradient (a tuple). In this method, a gradient-to-input backward propagation process is constructed.
In batch 0, the values at positions 3, 1, and 3 are randomly selected. The output gradient is 1, and the reverse gradient is `[0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 2.00000000e+000, 0.00000000e+000]`, which meets the expectation.

### Dynamic Shape Workarounds

Generally, dynamic shape is introduced due to the following reasons:

- The input shape is not fixed.
- Operators that cause shape changes exist during network execution.
- Different branches of the control flow introduce shape changes.

Now, let's look at some workarounds for these scenarios.

#### Input Shape Not Fixed

1. You can add pads to the input data to a fixed shape. For example, [Data Processing](https://gitee.com/mindspore/models/blob/master/official/audio/DeepSpeech2/src/dataset.py#L153) of deep_speechv2 specifies the maximum length of `input_length`. Short `input_length` are padded with 0s, and long `input_length` are randomly truncated. Note that this method may affect the training accuracy. Therefore, the training accuracy and training performance need to be balanced.

2. You can set a group of fixed input shapes to process the input into several fixed scales. For example, in [Data Processing](https://gitee.com/mindspore/models/blob/master/official/cv/YOLOv3/src/yolo_dataset.py#L177) of YOLOv3_darknet53, the processing function `multi_scale_trans` is added to the batch method, and a shape is randomly selected from [MultiScaleTrans](https://gitee.com/mindspore/models/blob/master/official/cv/YOLOv3/src/transforms.py#L456) for processing.

Currently, the support for completely random input shapes is limited and needs to be supported in the new version.

#### Operations that Cause Shape Changes During Network Execution

In the scenario where tensors with unfixed shapes are generated during network running, the most common method is to construct a mask to filter out values in invalid positions. For example, in the detection scenario, some boxes need to be selected based on the iou results of the prediction box and real box.
The PyTorch implementation is as follows:

```python
def box_select_torch(box, iou_score):
    mask = iou_score > 0.3
    return box[mask]
```

In versions later than MindSpore 1.8, `masked_select` is supported in all scenarios. In MindSpore, `masked_select` can be implemented as follows:

```python
import mindspore as ms
from mindspore import ops

ms.set_seed(1)

def box_select_ms(box, iou_score):
    mask = (iou_score > 0.3).expand_dims(1)
    return ops.masked_select(box, mask)
```

Let's look at the result comparison.

```python
import torch
import numpy as np
import mindspore as ms

ms.set_seed(1)

box = np.random.uniform(0, 1, (3, 4)).astype(np.float32)
iou_score = np.random.uniform(0, 1, (3,)).astype(np.float32)

print("box_select_ms", box_select_ms(ms.Tensor(box), ms.Tensor(iou_score)))
print("box_select_torch", box_select_torch(torch.from_numpy(box), torch.from_numpy(iou_score)))
```

```text
    box_select_ms [0.14675589 0.09233859 0.18626021 0.34556073]
    box_select_torch tensor([[0.1468, 0.0923, 0.1863, 0.3456]])
```

However, after this operation, dynamic shape is generated, which may cause problems in subsequent network calculation. Currently, you are advised to use the mask to avoid this problem.

```python
def box_select_ms2(box, iou_score):
    mask = (iou_score > 0.3).expand_dims(1)
    return box * mask, mask
```

In subsequent computation, if some box operations are involved, check whether the mask needs to be multiplied to filter invalid results.

If a tensor with an unfixed shape is obtained due to feature selection during loss computation, the processing method is basically the same as that during network running. The only difference is that the loss part may not have other operations and the mask does not need to be returned.

For example, we want to select the values of the first 70% positive samples to compute the loss.
The PyTorch implementation is as follows:

```python
import torch
import torch.nn as torch_nn

class ClassLoss_pt(torch_nn.Module):
    def __init__(self):
        super(ClassLoss_pt, self).__init__()
        self.con_loss = torch_nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label):
        mask = label > 0
        vaild_label = label * mask
        pos_num = torch.clamp(mask.sum() * 0.7, 1).int()
        con = self.con_loss(pred, vaild_label.long()) * mask
        loss, _ = torch.topk(con, k=pos_num)
        return loss.mean()
```

`torch.topk` is used to obtain the first 70% positive sample data. Currently, MindSpore does not support K as a variable. Therefore, you need to convert the method to obtain the Kth largest value and then obtain the mask of the top K based on the value. The MindSpore implementation is as follows:

```python
import mindspore as ms
from mindspore import ops
from mindspore import nn as ms_nn

class ClassLoss_ms(ms_nn.Cell):
    def __init__(self):
        super(ClassLoss_ms, self).__init__()
        self.con_loss = ms_nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.sort_descending = ops.Sort(descending=True)

    def construct(self, pred, label):
        mask = label > 0
        vaild_label = label * mask
        pos_num = ops.maximum(mask.sum() * 0.7, 1).astype(ms.int32)
        con = self.con_loss(pred, vaild_label.astype(ms.int32)) * mask
        con_sort, _ = self.sort_descending(con)
        con_k = con_sort[pos_num - 1]
        con_mask = (con >= con_k).astype(con.dtype)
        loss = con * con_mask
        return loss.sum() / con_mask.sum()
```

Let's look at the test result.

```python
import torch
import numpy as np
import mindspore as ms
ms.set_seed(1)

pred = np.random.uniform(0, 1, (5, 2)).astype(np.float32)
label = np.array([-1, 0, 1, 1, 0]).astype(np.int32)
print("pred", pred)
print("label", label)
t_loss = ClassLoss_pt()
cls_loss_pt = t_loss(torch.from_numpy(pred), torch.from_numpy(label))
print("cls_loss_pt", cls_loss_pt)
m_loss = ClassLoss_ms()
cls_loss_ms = m_loss(ms.Tensor(pred), ms.Tensor(label))
print("cls_loss_ms", cls_loss_ms)
```

```text
    pred [[4.17021990e-01 7.20324516e-01]
     [1.14374816e-04 3.02332580e-01]
     [1.46755889e-01 9.23385918e-02]
     [1.86260208e-01 3.45560730e-01]
     [3.96767467e-01 5.38816750e-01]]
    label [-1  0  1  1  0]
    cls_loss_pt tensor(0.7207)
    cls_loss_ms 0.7207259
```

#### Shape Changes Introduced by Different Branches of Control Flows

The following is an example in the model analysis and preparation section:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
cond = (x > 0.5).any()

if cond:
    y = ops.masked_select(x, x > 0.5)
else:
    y = ops.zeros_like(x)
print(x)
print(cond)
print(y)
```

```text
    [4.17021990e-01 7.20324516e-01 1.14374816e-04 3.02332580e-01
     1.46755889e-01 9.23385918e-02 1.86260208e-01 3.45560730e-01
     3.96767467e-01 5.38816750e-01]
    True
    [0.7203245  0.53881675]
```

In `cond=True` mode, the maximum shape is the same as x. According to the preceding mask adding method, the maximum shape can be written as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
cond = (x > 0.5).any()

if cond:
    mask = (x > 0.5).astype(x.dtype)
else:
    mask = ops.zeros_like(x)
y = x * mask
print(x)
print(cond)
print(y)
```

```text
    [4.17021990e-01 7.20324516e-01 1.14374816e-04 3.02332580e-01
     1.46755889e-01 9.23385918e-02 1.86260208e-01 3.45560730e-01
     3.96767467e-01 5.38816750e-01]
    True
    [0.         0.7203245  0.         0.         0.         0.
     0.         0.         0.         0.53881675]
```

Note that if y is subsequently involved in other calculations, it needs to be passed in mask to do filtering on the valid positions.
