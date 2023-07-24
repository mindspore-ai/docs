# Common Network Components

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_en/network_component.md)

## Overview

MindSpore encapsulates some common network components for network training, inference, gradient calculation, and data processing.

These network components can be directly used by users and are also used in more advanced encapsulation APIs such as `model.train` and `model.eval`.

The following describes three network components, `GradOperation`, `WithLossCell`, and `TrainOneStepCell`, in terms of functions, usage, and internal use.

## GradOperation

GradOperation is used to generate the gradient of the input function. The `get_all`, `get_by_list`, and `sens_param` parameters are used to control the gradient calculation method. For details, see [MindSpore API](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/ops/mindspore.ops.GradOperation.html)
The following is an example of using GradOperation:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
import mindspore.ops as ops


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
GradNetWrtX(Net())(x, y)
```

```text
Tensor(shape=[2, 3], dtype=Float32, value=
[[1.41000009e+000, 1.60000002e+000, 6.59999943e+000],
 [1.41000009e+000, 1.60000002e+000, 6.59999943e+000]])
```

The preceding example is used to calculate the gradient value of `Net` to x. You need to define the network `Net` as the input of `GradOperation`. The instance creates `GradNetWrtX` that contains the gradient operation. Calling `GradNetWrtX` transfers the network to `GradOperation` to generate a gradient function, and transfers the input data to the gradient function to return the final result.

All other components, such as `WithGradCell` and `TrainOneStepCell`, involved in gradient calculation use `GradOperation`.
You can view the internal implementation of these APIs.

## WithLossCell

`WithLossCell` is essentially a `Cell` that contains the loss function. To build `WithLossCell`, you need to define the network and loss function in advance.

The following uses an example to describe how to use this function. First, you need to build a network. The content is as follows:

```python
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = ops.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output
```

The following is an example of using `WithLossCell`. Define the network and loss functions, create a `WithLossCell`, and input the input data and label data. `WithLossCell` returns the calculation result based on the network and loss functions.

```python
data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.ones([32]).astype(np.int32))
net = LeNet()
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
loss = net_with_criterion(data, label)
print("+++++++++Loss+++++++++++++")
print(loss)
```

The following information is displayed:

```text
+++++++++Loss+++++++++++++
2.302585
```

## TrainOneStepCell

`TrainOneStepCell` is used to perform single-step training of the network and return the loss result after each training result.

The following describes how to build an instance for using the `TrainOneStepCell` API to perform network training. The import code of the `LeNet` and package name is the same as that in the previous case.

```python
data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.ones([32]).astype(np.int32))
net = LeNet()
learning_rate = 0.01
momentum = 0.9

optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
for i in range(5):
    train_network.set_train()
    res = train_network(data, label)
    print(f"+++++++++result:{i}++++++++++++")
    print(res)
```

```text
+++++++++result:0++++++++++++
2.302585
+++++++++result:1++++++++++++
2.2935712
+++++++++result:2++++++++++++
2.2764661
+++++++++result:3++++++++++++
2.2521412
+++++++++result:4++++++++++++
2.2214084
```

In the case, an optimizer and a `WithLossCell` instance are built, and then a training network is initialized in `TrainOneStepCell`. The case is repeated for five times, that is, the network is trained for five times, and the loss result of each time is output, the result shows that the loss value gradually decreases after each training.

The following content will describe how MindSpore uses more advanced encapsulation APIs, that is, the `train` method in the `Model` class to train a model. Many network components, such as `TrainOneStepCell` and `WithLossCell`, will be used in the internal implementation.
You can view the internal implementation of these components.
