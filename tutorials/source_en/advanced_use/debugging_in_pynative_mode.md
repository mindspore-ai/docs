# Debugging in PyNative Mode

<a href="https://gitee.com/mindspore/docs/blob/r0.3/tutorials/source_en/advanced_use/debugging_in_pynative_mode.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindSpore supports the following running modes which are optimized in terms of debugging or running:

- PyNative mode: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.
- Graph mode: static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution. This mode uses technologies such as graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

By default, MindSpore is in PyNative mode. You can switch it to the graph mode by calling `context.set_context(mode=context.GRAPH_MODE)`. Similarly, MindSpore in graph mode can be switched to the PyNative mode through `context.set_context(mode=context.PYNATIVE_MODE)`.

In PyNative mode, single operators, common functions, network inference, and separated gradient calculation can be executed. The following describes the usage and precautions.

## Executing a Single Operator

Execute a single operator and output the result, as shown in the following example.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
output = conv(input_data)
print(output.asnumpy())
```

Output:

```python
[[[[-0.02190447 -0.05208071 -0.05208071 -0.05208071 -0.06265172]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01430791 -0.04892948 -0.04892948 -0.04892948 -0.01096004]]

[[ 0.00802889 -0.00229866 -0.00229866 -0.00229866 -0.00471579]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01784375 0.01185635 0.01185635 0.01185635 0.01839031]]

[[ 0.04841832 0.03321705 0.03321705 0.03321705 0.0342317 ]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.04680437 0.03465693 0.03465693 0.03465693 0.00171057]]

[[-0.01783456 -0.00459451 -0.00459451 -0.00459451 0.02316688]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.05016355 0.03958241 0.03958241 0.03958241 0.03443141]]]]
```


## Executing a Common Function

Combine multiple operators into a function, call the function to execute the operators, and output the result, as shown in the following example:

**Example Code**
```python
import numpy as np
from mindspore import context, Tensor
from mindspore.ops import functional as F

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def tensor_add_func(x, y):
    z = F.tensor_add(x, y)
    z = F.tensor_add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = tensor_add_func(x, y)
print(output.asnumpy())
```

**Output**

```python
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

> Parallel execution and summary is not supported in PyNative mode, so parallel and summary related operators can not be used.


### Improving PyNative Performance

MindSpore provides the staging function to improve the execution speed of inference tasks in PyNative mode. This function compiles Python functions or Python class methods into computational graphs in PyNative mode and improves the execution speed by using graph optimization technologies, as shown in the following example:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops.operations as P
from mindspore.common.api import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = P.TensorAdd()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(x, y) # Staging mode
tensor_add = P.TensorAdd()
res = tensor_add(x, z) # PyNative mode
print(res.asnumpy())
```
**Output**

```python
[[3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]]
```

In the preceding code, the `ms_function` decorator is added before `construct` of the `TensorAddNet` class. The decorator compiles the `construct` method into a computational graph. After the input is given, the graph is delivered and executed, `tensor_add` in the preceding code is executed in the common PyNative mode.

It should be noted that, in a function to which the `ms_function` decorator is added, if an operator (such as `pooling` or `tensor_add`) that does not need parameter training is included, the operator can be directly called in the decorated function, as shown in the following example:

**Example Code**

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops.operations as P
from mindspore.common.api import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

tensor_add = P.TensorAdd()

@ms_function
def tensor_add_fn(x, y):
    res = tensor_add(x, y)
    return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
z = tensor_add_fn(x, y)
print(z.asnumpy())
```
**Output**

```shell
[[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

If the decorated function contains operators (such as `Convolution` and `BatchNorm`) that require parameter training, these operators must be instantiated before the decorated function is called, as shown in the following example:

**Example Code**

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common.api import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
conv_obj.init_parameters_data()
@ms_function
def conv_fn(x):
    res = conv_obj(x)
    return res

input_data = np.random.randn(2, 3, 6, 6).astype(np.float32)
z = conv_fn(Tensor(input_data))
print(z.asnumpy())
```

**Output**

```shell
[[[[ 0.10377571 -0.0182163 -0.05221086]
[ 0.1428334 -0.01216263 0.03171652]
[-0.00673915 -0.01216291 0.02872104]]

[[ 0.02906547 -0.02333629 -0.0358406 ]
[ 0.03805163 -0.00589525 0.04790922]
[-0.01307234 -0.00916951 0.02396654]]

[[ 0.01477884 -0.06549098 -0.01571796]
[ 0.00526886 -0.09617482 0.04676902]
[-0.02132788 -0.04203424 0.04523344]]

[[ 0.04590619 -0.00251453 -0.00782715]
[ 0.06099087 -0.03445276 0.00022781]
[ 0.0563223 -0.04832596 -0.00948266]]]

[[[ 0.08444098 -0.05898955 -0.039262 ]
[ 0.08322686 -0.0074796 0.0411371 ]
[-0.02319113 0.02128408 -0.01493311]]

[[ 0.02473745 -0.02558945 -0.0337843 ]
[-0.03617039 -0.05027632 -0.04603915]
[ 0.03672804 0.00507637 -0.08433761]]

[[ 0.09628943 0.01895323 -0.02196114]
[ 0.04779419 -0.0871575 0.0055248 ]
[-0.04382382 -0.00511185 -0.01168541]]

[[ 0.0534859 0.02526264 0.04755395]
[-0.03438103 -0.05877855 0.06530266]
[ 0.0377498 -0.06117418 0.00546303]]]]
```

## Debugging Network Train Model

In PyNative mode, the gradient can be calculated separately. As shown in the following example, `GradOperation` is used to calculate all input gradients of the function or the network.

**Example Code**

```python
from mindspore.ops import composite as C
import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def mul(x, y):
    return x * y

def mainf(x, y):
    return C.GradOperation('get_all', get_all=True)(mul)(x, y)

print(mainf(1,2))
```

**Output**

```python
(2, 1)
```

During network training, obtain the gradient, call the optimizer to optimize parameters (the breakpoint cannot be set during the reverse gradient calculation), and calculate the loss values. Then, network training is implemented in PyNative mode.

**Complete LeNet Sample Code**

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import Dense, WithLossCell, SoftmaxCrossEntropyWithLogits, Momentum

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return C.GradOperation('get_by_list', get_by_list=True)(self.network, weights)(x, label)

net = LeNet5()
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
net_with_criterion = WithLossCell(net, criterion)
train_network = GradWrap(net_with_criterion)
train_network.set_train()

input_data = Tensor(np.ones([net.batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.ones([net.batch_size]).astype(np.int32))
output = net(Tensor(input_data))
loss_output = criterion(output, label)
grads = train_network(input_data, label)
success = optimizer(grads)
loss = loss_output.asnumpy()
print(loss)
```

**Output**

```python
2.3050091
```

In the preceding execution, an intermediate result of network execution can be obtained at any required place in construct function, and the network can be debugged by using the Python Debugger (pdb).
