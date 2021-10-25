# 网络内构造常量

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [网络内构造常量](#网络内构造常量)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/constexpr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

`mindspore.ops.constexpr`中提供了一个@constexpr的Python 装饰器，该装饰器可以用于修饰一个函数，该函数在编译阶段将会通过Python解释器执行，最终在MindSpore的类型推导阶段被常量折叠成为ANF图的一个常量节点(ValueNode)。

由于该函数在MindSpore编译时期进行，所以使用@constexpr函数时，要求输入函数的入参必须为一个编译时刻就能够确定的常量值，否则如果该函数入参为一个编译时刻无法确定的值，那么入参将会为None，从而可能导致函数输出与预期不符。

当@constexpr的入参为提前明确的参数时可以实现一些在construct函数中不支持的操作。比如根据shape创建Tensor等。

为了避免出现@constexpr输入为编译时无法确定的值，可以在内部进行对None的判断处理，避免一些未知错误。

代码样例如下:

```python
import numpy as np
from mindspore.ops import constexpr
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor

@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()
    def construct(self, x):
        return self.relu(construct_tensor(ops.shape(x)))

net = Net()
x = Tensor(np.random.random([7,6,3]))
out = net(x)
print(out)
```

运行结果如下:

```text
[7 6 3]
```

如下所示，如果我们将Net改成输入为编译时无法确定的值时，则会抛出异常。 由于construct_tensor输入为运行ReLU时才能确定的值。在constexpr中会抛出ValueError。

```python
@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()
    def construct(self, x):
        return self.relu(construct_tensor(self.relu(x)))

net = Net()
x = Tensor(np.random.random([7,6,3]))
out = net(x)
print(out)
```

运行结果如下:

```text
ValueError: input is an unknown value
```
