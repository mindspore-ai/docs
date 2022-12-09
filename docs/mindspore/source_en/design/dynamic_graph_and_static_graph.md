# Combination of Dynamic and Static Graphs

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/dynamic_graph_and_static_graph.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## The Concept of Static and Dynamic Graphs

There are two execution modes of the mainstream deep learning frameworks, namely static graph mode and dynamic graph mode.

In static graph mode, the program firstly generates the graph structure of the neural network, then executes the computational operations involved in the graph during compilation execution. Therefore, in static graph mode, the compiler uses techniques such as graph optimization to optimize the execution graph to a greater extent, resulting in better execution performance that helps scale deployment and cross-platform operation.

In dynamic graph mode, the program is executed in the order in which the code is written, and the reverse execution graph is dynamically generated during the execution of the forward process based on the principle of backward propagation. In this mode, the compiler sends down the individual operators in the neural network for execution one by one, making it easy for the user to write and debug the neural network model.\

## MindSpore Static Graph

In MindSpore, the static graph mode, also known as Graph mode, can be set to static graph mode by `set_context(mode=GRAPH_MODE)`. Static graph mode is more suitable for scenarios where the network is fixed and high performance is required. In static graph mode, the compiler can perform global optimization for the graph based on techniques such as graph optimization, whole graph offloading of computational graph. Therefore, better performance can be obtained under static graph, but the execution graph is converted from the source code. Not all Python syntax is supported under static graphs. For the detailed information, please refer to [Syntax Support](https://www.mindspore.cn/docs/en/master/note/syntax_list.html).

### Graph Mode Execution Principle

In Graph mode, MindSpore converts Python source code into IR by means of source code conversion, then performs relevant graph optimization based on this, and finally executes the optimized graph on hardware devices. MindSpore uses a functional IR based on graph representation, i.e. MindIR, by using a semantics close to the ANF functional style. The Graph mode is compiled and optimized based on MindIR. To use the Graph mode, you need to use the `nn.Cell` class and write the execution code in the `construct` function or call the `@jit` decorator.

An code example for the Graph model is shown below:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

net = Net()
print(net(x, y))
```

```text
[ 4. 10. 18.]
```

### Graph Mode Auto-differentiation Principle

In MindSpore, the principle of auto-differentiation in Graph mode can be found in [Auto-differentiation](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html).

## MindSpore Dynamic Graph

In MindSpore, dynamic graph mode is also known as PyNative mode, which can be set to dynamic graph mode by `set_context(mode=PYNATIVE_MODE)`. In script development and network flow debugging, it is recommended to use dynamic graph mode for debugging, which supports the execution of single operators, common functions and networks, and separate gradient solving operations.

### PyNative Mode Execution Principle

In PyNative mode, users can use the full Python API. In addition, for using the API provided by MindSpore, the framework will execute the operations of the operator API on the corresponding hardware platform according to the hardware platform (Ascend, GPU, CPU) selected by the user and return the corresponding results. The overall execution process of the framework is as follows:

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/framework.png)

Through the front-end Python API, call to the framework layer, and finally to the corresponding hardware devices to perform calculations. For example, to complete an addition

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
output = ops.add(x, y)
print(output.asnumpy())
```

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

In this example, when the Python interface ops.add(x, y) is called, the Python interface call is called to the C++ layer of the framework via Pybind11, and converted to C++ call. Then the framework will select the corresponding hardware device according to the device_target set by the users, and execute the add operation on that hardware device.

From the above principle, we can see that in PyNative mode, Python script code will be executed according to Python syntax, and the execution process involves MindSpore's API, which will be accelerated by executing on different hardware according to user settings. Therefore, in PyNative mode, users can use Python syntax and debugging methods at will, for example, you can use common IDEs such as PyCharm and VS Code to debug code.

### PyNative Mode Auto-differentiation Principle

In the previous introduction, we can see that the execution of the forward procedure under PyNative is performed exactly according to Python syntax. Under PyNative, backward propagation is implemented based on Tensor. We record all the operations applied to Tensor during the execution of the forward process, and for each operation find its reverse, and string all the reverse processes together to form the overall backward propagation graph (reverse graph for short). Eventually, the reverse graph is executed on the device to calculate the gradient.

The following code is an example of the reverse composition process: multiply the matrix x with a fixed parameter z, then perform matrix multiplication with y, and finally derives x.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([2.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net

    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
        return gradient_function(x, y)

x = ms.Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=ms.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
[[9.02      5.4       7.2000003]
 [9.02      5.4       7.2000003]]
```

According to the above composition principle under PyNative, it can be seen that in the forward propagation process, we record the calculation process of Mul. According to the definition of reverse bprop corresponding to the Mul, we get the reverse MulGrad operator. According to the definition of Mul operator's bprop, as follows:

```python
from mindspore.ops._grad.grad_base import bprop_getters

@bprop_getters.register(ops.Mul)
def get_bprop_mul(self):
    """Grad definition for `Mul` operation."""
    mul_func = P.Mul()

    def bprop(x, y, out, dout):
        bc_dx = mul_func(y, dout)
        bc_dy = mul_func(x, dout)
        return binop_grad_common(x, y, bc_dx, bc_dy)

    return bprop
```

It can be seen that reversing the input to Mul requires backward propagation gradient values of two input and output, at which point z can be connected to MulGrad based on the actual input values. And so on, for the next operator Matmul, the MatmulGrad information is obtained accordingly, and then the contextual gradient propagation is connected according to the input and output of bprop.

Similarly for the input y derivation, the same procedure can be used for the derivation.

### Control flow in PyNative Mode

In the PyNative mode, scripts are executed according to the Python syntax, so in MindSpore, there is no special treatment for the control flow syntax, which is directly expanded and executed according to the Python syntax, and automatic differentiation is performed on the expanded execution operator. For example, for a for loop, the statements in the for loop are continuously executed under PyNative and automatic differentiation is performed on the operators according to the specific number of loops.

## Dynamic and Static Unification

### Overview

The industry currently supports both dynamic and static graph modes. Dynamic graphs are executed by interpretation, with dynamic syntax affinity and flexible expression, and static graphs are executed by using jit compilation optimization, more inclined to static syntax and more restrictions in syntax. For dynamic and static graph modes, firstly MindSpore unifies the API expression, uses the same API in both modes, secondly, unifies the underlying differential mechanism of dynamic and static graphs.

### Interconversion of Dynamic and Static Graphs

In MindSpore, we can switch the execution between using dynamic or static graphs by controlling the mode input parameters. For example:

```python
ms.set_context(mode=ms.PYNATIVE_MODE)
```

Since there are restrictions on Python syntax under static graphs, switching from dynamic to static graphs requires compliance with the syntax restrictions of static graphs in order to execute correctly by using static graphs. For more syntax restrictions for static graphs, refer to [Static Graph Syntax Restrictions](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Combination of Static and Dynamic

MindSpore supports mixed execution by using static compilation under dynamic graphs. The function objects that need to be executed with static graphs by using jit modification, and in this way you can achieve mixed execution of dynamic and static graphs. For more use of jit, refer to [jit documentation](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html#just-in-time-compilation).

For example:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class AddMulMul(nn.Cell):
    def __init__(self):
        super(AddMulMul, self).__init__()
        self.param = ms.Parameter(ms.Tensor(0.5, ms.float32))

    @ms.jit
    def construct(self, x):
        x = x + x
        x = x * self.param
        x = x * x
        return x

class CellCallSingleCell(nn.Cell):
    def __init__(self):
        super(CellCallSingleCell, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()
        self.add_mul_mul = AddMulMul()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.add_mul_mul(x)
        x = self.relu(x)
        return x

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
inputs = ms.Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
net = CellCallSingleCell()
out = net(inputs)
print(out)
```

```text
[[[[15.99984]]

  [[15.99984]]]]
```

### JIT Fallback

JIT Fallback is a functional feature proposed to realize the unification of dynamic and static. With features such as JIT Fallback, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax usage experience close to that of dynamic graphs.

JIT Fallback achieves the unification of static and dynamic graphs from the perspective of static graphs. In MindSpore static diagram mode, users need to follow MindSpore [static diagram syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) when writing programs. Constraints exist on the use of the syntax. In dynamic graph mode, Python script code is executed according to the Python syntax, and users can use any Python syntax. It can be seen that the syntax constraint restrictions are different for static and dynamic graphs. The JIT Fallback feature allows static graphs to support as many dynamic diagram syntaxes as possible, allowing users to switch between static and dynamic graphs in a flexible manner.

The current JIT Fallback supports some constant scenarios of static graph mode, including calling third-party libraries, creating and using Tensor, and calling Python's print printing. For more instructions and use of JIT Fallback, please refer to [JIT Fallback documentation](https://www.mindspore.cn/docs/en/master/design/jit_fallback.html).

The code example is as follows, where the MindSpore static graph mode does not support calling NumPy third-party libraries and creating Tensor objects in construct, so `x = np.array([1, 2, 3])` and `y = Tensor(x)` in the use case will be supported by the JIT Fallback feature.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self):
        x = np.array([1, 2, 3])
        y = ms.Tensor(x)
        return y

net = Net()
print(net())
```

```text
[1 2 3]
```
