# 使用PyNative模式调试

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_zh_cn/advanced_use/debug_in_pynative_mode.md" target="_blank"><img src="../_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.2/mindspore_debugging_in_pynative_mode.ipynb"><img src="../_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9kZWJ1Z2dpbmdfaW5fcHluYXRpdmVfbW9kZS5pcHluYg==&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="../_static/logo_modelarts.png"></a>

## 概述

MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化:

- PyNative模式：也称动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。
- Graph模式：也称静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。

默认情况下，MindSpore处于PyNative模式，可以通过`context.set_context(mode=context.GRAPH_MODE)`切换为Graph模式；同样地，MindSpore处于Graph模式时，可以通过 `context.set_context(mode=context.PYNATIVE_MODE)`切换为PyNative模式。

PyNative模式下，支持执行单算子、普通函数和网络，以及单独求梯度的操作。下面将详细介绍使用方法和注意事项。

> PyNative模式下为了提升性能，算子在device上使用了异步执行方式，因此在算子执行错误的时候，错误信息可能会在程序执行到最后才显示。

## 执行单算子

执行单个算子，并打印相关结果，如下例所示。

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

输出：

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

## 执行普通函数

将若干算子组合成一个函数，然后直接通过函数调用的方式执行这些算子，并打印相关结果，如下例所示。

示例代码：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def add_func(x, y):
    z = ops.add(x, y)
    z = ops.add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = add_func(x, y)
print(output.asnumpy())
```

输出：

```python
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

> PyNative不支持summary功能，图模式summary相关算子不能使用。

### 提升PyNative性能

为了提高PyNative模式下的前向计算任务执行速度，MindSpore提供了Staging功能，该功能可以在PyNative模式下将Python函数或者Python类的方法编译成计算图，通过图优化等技术提高运行速度，如下例所示。

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(x, y) # Staging mode
add = ops.Add()
res = add(x, z) # PyNative mode
print(res.asnumpy())
```

输出：

```python
[[3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]]
```

上述示例代码中，在`TensorAddNet`类的`construct`之前加装了`ms_function`装饰器，该装饰器会将`construct`方法编译成计算图，在给定输入之后，以图的形式下发执行，而上一示例代码中的`add`会直接以普通的PyNative的方式执行。

需要说明的是，加装了`ms_function`装饰器的函数中，如果包含不需要进行参数训练的算子（如`pooling`、`add`等算子），则这些算子可以在被装饰的函数中直接调用，如下例所示。

示例代码：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

add = ops.Add()

@ms_function
def add_fn(x, y):
    res = add(x, y)
    return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
z = add_fn(x, y)
print(z.asnumpy())
```

输出：

```shell
[[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

如果被装饰的函数中包含了需要进行参数训练的算子（如`Convolution`、`BatchNorm`等算子），则这些算子必须在被装饰等函数之外完成实例化操作，如下例所示。

示例代码：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore import ms_function

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

输出：

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

## 调试网络训练模型

PyNative模式下，还可以支持单独求梯度的操作。如下例所示，可通过`GradOperation`求该函数或者网络所有的输入梯度。需要注意，输入类型仅支持Tensor。

示例代码：

```python
import mindspore.ops as ops
import mindspore.context as context
from mindspore import dtype as mstype
from mindspore import Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def mul(x, y):
    return x * y

def mainf(x, y):
    return ops.GradOperation(get_all=True)(mul)(x, y)

print(mainf(Tensor(1, mstype.int32), Tensor(2, mstype.int32)))
```

输出：

```python
(Tensor(shape=[], dtype=Int32, value=2), Tensor(shape=[], dtype=Int32, value=1))
```

在进行网络训练时，求得梯度然后调用优化器对参数进行优化（暂不支持在反向计算梯度的过程中设置断点），然后再利用前向计算loss，从而实现在PyNative模式下进行网络训练。

完整LeNet示例代码：

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
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
        self.reshape = ops.Reshape()

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
        return ops.GradOperation(get_by_list=True)(self.network, weights)(x, label)

net = LeNet5()
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
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

输出：

```python
2.3050091
```

上述执行方式中，可以在`construct`函数任意需要的地方设置断点，获取网络执行的中间结果，通过pdb的方式对网络进行调试。
