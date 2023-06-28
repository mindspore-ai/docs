# 常用网络组件

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/network_component.md" target="_blank"><img src="./_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_network_component.ipynb"><img src="./_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV9uZXR3b3JrX2NvbXBvbmVudC5pcHluYg==&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="./_static/logo_modelarts.png"></a>

## 概述

MindSpore封装了一些常用的网络组件，用于网络的训练、推理、求梯度和数据处理等操作。

这些网络组件可以直接被用户使用，同样也会在`model.train`和`model.eval`等更高级的封装接口内部进行使用。

本节内容将会介绍三个网络组件，分别是`GradOperation`、`WithLossCell`和`TrainOneStepCell`，将会从功能、用户使用和内部使用三个方面来进行介绍。

## GradOperation

GradOperation组件用于生成输入函数的梯度，利用`get_all`、`get_by_list`和`sens_param`参数控制梯度的计算方式，细节内容详见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/ops/mindspore.ops.GradOperation.html)。

GradOperation的使用实例如下：

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
        self.z = Parameter(Tensor(np.array([1.0], np.float32)))
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

上面的例子是计算`Net`相对与x的梯度值，首先需要定义网络`Net`作为`GradOperation`的输入，实例创建了包含梯度运算的`GradNetWrtX`。调用`GradNetWrtX`是将网络传入`GradOperation`生成梯度函数，将输入数据传入梯度函数中返回最终结果。

输出如下：

```text
[[1.4100001 1.5999999 6.6      ]
 [1.4100001 1.5999999 6.6      ]]
```

MindSpore涉及梯度计算的其他组件，例如`WithGradCell`和`TrainOneStepCell`等，都用到了`GradOperation`，
感兴趣的读者可以查看这些接口的内部实现。

## WithLossCell

`WithLossCell`本质上是一个包含损失函数的`Cell`，构造`WithLossCell`需要事先定义好网络和损失函数。

下面通过一个实例来介绍其具体的使用， 首先需要构造一个网络，内容如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


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

下面是`WithLossCell`的使用实例，分别定义好网络和损失函数，然后创建一个`WithLossCell`，传入输入数据和标签数据，`WithLossCell`内部根据网络和损失函数返回计算结果。

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

输出如下：

```text
+++++++++Loss+++++++++++++
2.302585
```

## TrainOneStepCell

`TrainOneStepCell`功能是执行网络的单步训练，返回每次训练结果后的loss结果。

下面构造一个使用`TrainOneStepCell`接口进行网络训练的实例，其中`LeNet`和包名的导入代码和上个用例共用。

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

用例中构造了优化器和一个`WithLossCell`的实例，然后传入`TrainOneStepCell`中初始化一个训练网络，用例循环五次，相当于网络训练了五次，并输出每次的loss结果，由结果可以看出每次训练后loss值在逐渐减小。

输出如下：

```text
+++++++++result:0++++++++++++
2.302585
+++++++++result:1++++++++++++
2.2935824
+++++++++result:2++++++++++++
2.2765071
+++++++++result:3++++++++++++
2.2522228
+++++++++result:4++++++++++++
2.2215357
```

后续内容会介绍MindSpore使用更加高级封装的接口，即`Model`类中的`train`方法训练模型，在其内部实现中会用到
`TrainOneStepCell`和`WithLossCell` 等许多网络组件，感兴趣的读者可以查看其内部实现。
