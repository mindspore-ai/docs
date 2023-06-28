# Enabling Graph Kernel Fusion

`Linux` `GPU` `Model Optimization` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/enable_graph_kernel_fusion.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Introduction

The graph kernel fusion is used to optimize network performance by cooperating with JIT operator compilation. With analyzing and evaluating the compute graph, it will apply optimization such as computing workload reduction, operator splitting, fusion and special operator compiling, to reduce network execution time. Also, the whole optimization process is completed automatically only if the graph kernel setting is enabled. This will help the user focus on the network development.

The graph kernel fusion is available for:

- Network with high performance requirement;
- Custom combination operators with high performance requirement.

## Enabling Method

The graph kernel is disabled by default. We can just specify the `enable_graph_kernel=True` parameter for `context` in the training script to enable it.

```python
from mindspore import context
context.set_context(enable_graph_kernel=True)
```

> Only Graph Mode is supported by graph kernel.

### Sample Scripts

To illustrate the fusion scenario, we construct a simple network `MyNet`, including multiplication and addition operators. The two operators will be fused together with enabled graph kernel:

```python
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# save graph ir to view fusion detail.
context.set_context(save_graphs=True)
# enable graph kernel optimization.
context.set_context(enable_graph_kernel=True)

class MyNet(Cell):
    def __init__(self):
        super(MyNet, self).__init__()
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x):
        a = self.mul(x, 2.0)
        res = self.add(a, 1.0)
        return res

x = np.ones((4, 4)).astype(np.float32) * 0.5
net = MyNet()
result = net(Tensor(x))
print("result: {}".format(result))
```

The output is:

```text
result: [[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

The fusion of this graph is shown in Figure 1, the left graph is without graph kernel fusion being enabled and the right one is with graph kernel fusion being enabled, which can be checked by dumped graph IR or device profiling.

![fuse basic example](images/graph_kernel_example_fuse_basic.png)

Figure 1 Graph kernel fusion on computational graph

## Custom Combination Operators

We can easily implement high-performance custom combination operators based on graph kernel. The steps are as follows:

1. Define custom operator by combining basic operators;
2. Enable Graph Kernel;
3. Graph kernel automatically fuses the basic operators and generates high-performance fusion operators.

### Sample Scripts

We construct a simple network `MyNet` and define the custom operator `MyOp`:

```python
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# enable graph kernel optimization.
context.set_context(enable_graph_kernel=True)

class MyOp(Cell):
    """ my first custom OP composited by basic OPs """
    def __init__(self):
        super(MyOp, self).__init__()
        self.sub = P.Sub()
        self.mul = P.Mul()

    def construct(self, x, y):
        a = self.sub(x, y)
        return self.mul(a, x)

class MyNet(Cell):
    def __init__(self):
        super(MyNet, self).__init__()
        self.mul = P.Mul()
        self.pow = P.Pow()
        self.my_op = MyOp()

    def construct(self, x, y):
        a = self.mul(x, 2.0)
        b = self.pow(a, 3.0)
        res = self.my_op(b, y)
        return res

x = np.ones((4, 4)).astype(np.float32) * 0.2
y = np.ones((4, 4)).astype(np.float32) * 0.3
net = MyNet()
result = net(Tensor(x), Tensor(y))
print("result: {}".format(result))
```

The output is:

```text
result: [[-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]
 [-0.015104 -0.015104 -0.015104 -0.015104]]
```

The fusion of this graph is shown in Figure 2, the left graph is without graph kernel fusion being enabled and the right one is with graph kernel fusion being enabled, which can be checked by dumped graph IR or device profiling.

![cusom op example](images/graph_kernel_example_custom_op.png)

Figure 2 Custom combination operator on computational graph
