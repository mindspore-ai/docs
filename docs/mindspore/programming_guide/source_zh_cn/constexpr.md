# contexpr

<!-- TOC -->

- [constexpr](#constexpr)
    - [概述](#概述)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/constexpr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

`mindspore.ops.constexpr`中提供了一个@constexpr的Python 装饰器，该装饰器可以用于修饰一个函数，该函数在编译阶段将会通过Python解释器执行，最终在MindSpore的类型推导阶段被常量折叠成为ANF图的一个常量节点(ValueNode)。

由于该函数在MindSpore编译时期进行，所以使用@constexpr函数时，要求输入函数的入参必须为一个编译时刻就能够确定的常量值，否则如果该函数入参为一个编译时刻无法确定的值，那么入参将会为None，从而可能导致函数输出与预期不符。

代码样例如下:

```python
import numpy as np
from mindspore.ops import constexpr
from mindspore.ops import ops
import mindspore.nn as nn

@constexpr
def construct_tensor(x):
    return Tensor(np.array(x))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()
    def construct(self):
        return self.relu(construct_tensor(x))

net = Net()
out = net()
print(out)
```

运行结果如下:

```text

[1 2 0 4]
```