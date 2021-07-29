# WithLossCell

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/withlosscell.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

`WithLossCell` is essentially a `Cell` that contains the loss function. To build `WithLossCell`, you need to define the network and loss function in advance.

The following uses an example to describe how to use this function. First, you need to build a network. The content is as follows:

```python
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn import Momentum
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
