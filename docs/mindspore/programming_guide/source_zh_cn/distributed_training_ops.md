# 分布式集合通信原语

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [分布式集合通信原语](#分布式集合通信原语)
    - [AllReduce](#allreduce)
    - [AllGather](#allgather)
    - [ReduceScatter](#reducescatter)
    - [Broadcast](#broadcast)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/distributed_training_ops.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

在分布式训练中涉及通信操作例如`AllReduce`、`ReduceScatter`、`AllGather`和`Broadcast`等操作，如下图所示：

![image](./images/communication.png)

## AllReduce

`AllReduce`操作会将每卡对应输入`tensor`进行求和操作，最终每张卡输出是相同的`tensor`，例如上图左上部分所示，每张卡各自的输入为`0, 1, 2, 3`，经过`AllReduce`之后，每张卡输出的结果为每张卡输入之和为6(0+1+2+3)。

```python
from mindspore.communication import init
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM, group="nccl_world_group")

    def construct(self, x):
        return self.allreduce_sum(x)

input_ = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_)
print(output)
[[4. 5. 6. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0.]]
```

## AllGather

`AllGather`操作会将每张卡的输入在第0维度上进行拼接，最终每张卡输出是相同的`tensor`。例如上图右上部分所示，每卡的输入是大小为1x1的`tensor`，经过`AllGather`操作之后，每卡的输入都构成了输出的一部分，对应的输出shape为[4,1]。其中索引为[0,0]元素值来自于`rank0`的输入，索引为[1,0]的元素值来自于`rank1`的输入。

```python
# This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.communication import init
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()

    def construct(self, x):
        return self.allgather(x)

input_x = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_x)
print(output)
[[1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]]
```

## ReduceScatter

`ReduceScatter`操作会将每张卡的输入先进行求和(`Reduce`)，然后在第0为维度按卡数切分，分发到对应的卡上。例如上图右下角所示，每卡的输入均为4x1的`tensor`，先进行求和得到[0,4, 8, 12]的`tensor`，然后进行分发，每卡获得1x1大小的`tensor`。

```python
# This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn
from mindspore import Tensor, context
from mindspore.communication import init
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

context.set_context(mode=context.GRAPH_MODE)
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.reducescatter = ops.ReduceScatter(ops.ReduceOp.SUM)

    def construct(self, x):
        return self.reducescatter(x)

input_ = Tensor(np.ones([8, 8]).astype(np.float32))
net = Net()
output = net(input_)
print(output)
[[2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2.]
 [2. 2. 2. 2. 2. 2. 2. 2.]]
```

## Broadcast

`Broadcast`操作是将某张卡的输入广播到其他卡上，常见于参数的初始化。例如将0卡大小为1x1的`tensor`进行广播，最终每张卡的结果均为相同的[[0]]。

```python
# This example should be run with multiple processes.
# Please refer to the tutorial > Distributed Training on mindspore.cn.
from mindspore import Tensor
from mindspore import context
from mindspore.communication import init
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

context.set_context(mode=context.GRAPH_MODE)
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.broadcast = ops.Broadcast(1)

    def construct(self, x):
        return self.broadcast((x,))

input_x = Tensor(np.ones([2, 4]).astype(np.int32))
net = Net()
output = net(input_x)
print(output)
[[1, 1, 1, 1],
 [1, 1, 1, 1]]]
```
