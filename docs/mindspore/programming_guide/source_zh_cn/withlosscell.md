# WithLossCell

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/withlosscell.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

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

```text
+++++++++Loss+++++++++++++
2.302585
```
