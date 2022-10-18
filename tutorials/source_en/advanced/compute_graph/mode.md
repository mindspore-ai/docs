# Dynamic and Static Graphs

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/compute_graph/mode.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (Graph) and a dynamic graph mode (PyNative).

- In static graph mode, when the program is built and executed, the graph structure of the neural network is generated first, and then the computation operations involved in the graph are performed. Therefore, in static graph mode, the compiler can achieve better execution performance by using technologies such as graph optimization, which facilitates large-scale deployment and cross-platform running.

- In dynamic graph mode, the program is executed line by line according to the code writing sequence. In the forward execution process, the backward execution graph is dynamically generated according to the backward propagation principle. In this mode, the compiler delivers the operators in the neural network to the device one by one for computing, facilitating users to build and debug the neural network model.

## Introduction to Dynamic and Static Graphs

MindSpore provides a unified encoding mode for static and dynamic graphs, significantly enhancing compatibility between both types of graphs. This enables you to switch between the static and dynamic graph modes by changing only one line of code, eliminating the need to develop multiple sets of code. The dynamic graph mode is the default mode of MindSpore and is mainly used for debugging, and the static graph mode has more efficient execution performance and is mainly used for deployment.

> When switching the running mode from dynamic graph to static graph, pay attention to the [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Mode Selection

You can configure the `context` parameter to control the program running mode. The differences between the dynamic graph mode and the static graph mode are as follows:

- **Application scenario:** The network structure of a static graph needs to be built at the beginning, and then the framework optimizes and executes the entire graph. This mode is applicable to scenarios where the network is fixed and high performance is required. Operators are executed line by line on a dynamic graph. Single operator, common functions, and networks can be executed, and gradients can be computed separately.

- **Network execution:** When the same network and operator are executed in static graph mode and dynamic graph mode, the accuracy effect is the same. The static graph mode uses technologies such as graph optimization and entire computational graph offloading. The static graph mode has higher network performance and efficiency, while the dynamic graph mode facilitates debugging and optimization.

- **Code debugging:** The dynamic graph mode is recommended for script development and network process debugging. In dynamic graph mode, you can easily set breakpoints and obtain intermediate results of network execution. You can also debug the network in pdb mode. In static graph mode, breakpoints cannot be set. You can only specify an operator for printing and view the output result after the network execution is complete.

### Mode Switching

During mode switching, you need to set the running mode in the context. Define the network model `MyNet` and the data used in subsequent code snippets for subsequent switching and display of the dynamic and static graph modes.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class MyNet(nn.Cell):
    """Customize the network to implement the addition of two tensors."""
    def __init__(self):
        super(MyNet, self).__init__()
        self.add = ops.Add()

    def construct(self, x, y):
        return self.add(x, y)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
```

Set the running mode to static graph mode.

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

net = MyNet()
print(net(x, y))
```

```text
    [5. 7. 9.]
```

When MindSpore is in static graph mode, you can switch to the dynamic graph mode by setting `mode=ms.PYNATIVE_MODE`. Similarly, when MindSpore is in dynamic graph mode, you can switch to the static graph mode by setting`mode=ms.GRAPH_MODE`. Pay attention to [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

```python
ms.set_context(mode=ms.PYNATIVE_MODE)

net = MyNet()
print(net(x, y))
```

```text
    [5. 7. 9.]
```

## Static Graph

In MindSpore, the static graph mode is also called the Graph mode, which is applicable to scenarios where the network is fixed and high performance is required. You can set the input parameter `mode` to `GRAPH_MODE` in the `set_context` API to set the static graph mode.

In static graph mode, the compiler can perform global optimization on graphs based on technologies such as graph optimization and entire computational graph offloading. Therefore, good performance can be obtained when the compiler is executed in static graph mode. However, the execution graph is converted from the source code. Therefore, not all Python syntax is supported in static graph mode. There are some special constraints. For details about the support, see [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Execution Principle of the Static Graph Mode

In static graph mode, MindSpore converts the Python source code into an intermediate representation (IR), optimizes the IR graph based on the IR, and executes the optimized graph on the hardware device.

MindSpore uses a graph-based functional IR called MindIR. The static graph mode is built and optimized based on MindIR. When using the static graph mode, you need to use the [nn.Cell](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell) class and writ execution code in the `construct` function.

### Code Example in Static Graph Mode

The code of the static graph mode is as follows. The neural network model implements the computation operation of $f(x, y)=x*y$.

```python
# Set the running mode to static graph mode.
ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    """Customize the network to implement the multiplication of two tensors."""
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        """Define execution code.""
        return self.mul(x, y)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

net = Net()

print(net(x, y))
```

```text
    [ 4. 10. 18.]
```

### Control Flow in Static Graph Mode

For details about control flows in static graph mode, see [Process Control Statements](https://mindspore.cn/tutorials/experts/en/master/network/control_flow.html).

## Dynamic Graph

In MindSpore, the dynamic graph mode is also called the PyNative mode. You can set the input parameter `mode` to `PYNATIVE_MODE` in the `set_context` API to set the dynamic graph mode.

During script development and network process debugging, you are advised to use the dynamic graph mode for debugging. The dynamic graph mode supports single-operator execution, common function execution, network execution, and independent gradient computation.

### Execution Principle of the Dynamic Graph Mode

In dynamic graph mode, you can use complete Python APIs. In addition, when APIs provided by MindSpore are used, the framework executes operator API operations on the corresponding hardware platform based on the selected hardware platform (Ascend/GPU/CPU) or environment information, and returns the corresponding result.

The overall execution process of the framework is as follows:

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/advanced/compute_graph/images/framework2.png)

The front-end Python API is called to the framework layer and finally computed on the corresponding hardware device.

The following uses the `ops.mul` operator to replace the network model that needs to be defined in static graph mode to implement the computation of $f(x, y)=x*y$.

```python
# Set the running mode to dynamic graph mode.
ms.set_context(mode=ms.PYNATIVE_MODE)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

output = ops.mul(x, y)

print(output.asnumpy())
```

```text
    [ 4. 10. 18.]
```

In the preceding sample code, when the `ops.mul(x, y)` API is called, the Python API at the MindSpore expression layer is called to the C++ layer of the MindSpore framework using [Pybind11](https://pybind11.readthedocs.io/en/stable/basics.html) and converted into the C++ API. Then, the framework automatically selects the corresponding hardware device based on the MindSpore installation environment information and performs the add operation on the hardware device.

According to the preceding principles, in PyNative mode, Python script code is executed based on the Python syntax. During the execution, Python APIs at the MindSpore expression layer are executed on different hardware based on user settings to accelerate performance.

Therefore, in dynamic graph mode, you can use Python syntax and debugging methods as required.

### Principle of Automatic Differentiation in Dynamic Graph Mode

In a dynamic graph, the forward propagation process is executed based on the Python syntax, and the backward propagation process is implemented based on tensors.

Therefore, during the forward propagation process, all operations applied to tensors are recorded and computed backward, and all backward propagation processes are connected to form an overall backward propagation graph. Finally, the backward graph is executed on the device and the gradient is computed.

The following uses a simple sample code to describe the principle of automatic differentiation in dynamic graph mode. Multiply the matrix x by a fixed parameter z, and then perform matrix multiplication with y:

$$f(x, y)=(x * z) * y \tag{1}$$

The sample code is as follows:

```python
# Set the running mode to dynamic graph mode.
ms.set_context(mode=ms.PYNATIVE_MODE)

class Net(nn.Cell):
    """Customize a network."""
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        x = self.matmul(x, y)
        return x

class GradNetWrtX(nn.Cell):
    """Define the derivation of x."""
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
    [[4.5099998 2.7       3.6000001]
     [4.5099998 2.7       3.6000001]]
```

> The accuracy may vary depending on the computing platform. Therefore, the execution results of the preceding code vary slightly on different platforms. For details about the derivation of the formula and the explanation of the preceding printed results, see [Automatic Derivation](https://www.mindspore.cn/tutorials/en/master/advanced/derivation.html#).

![forward](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/compute_graph/images/forward_backward.png)

It can be learned from the preceding dynamic graph mode that, in a forward propagation process, MindSpore records a computation process of Mul, and a backward MulGrad operator is obtained according to a definition of a backward bprop corresponding to Mul.

The bprop definition of the Mul operator is as follows:

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

You can see that two backward propagation gradient values of the input and output are required to compute the backward input of Mul. In this case, you can connect z to MulGrad based on the actual input value. The rest can be deduced by analogy. For the next operator Matmul, the MatmulGrad information is obtained accordingly, and then the context gradient propagation is connected based on the input and output of bprop.

Similarly, for derivation of input y, the same process may be used for derivation.

### Control Flow in Dynamic Graph Mode

In MindSpore, the control flow syntax is not specially processed. Instead, the control flow syntax is directly executed based on the Python syntax, and automatic differentiation operations are performed on the expanded execution operators.

For example, in a for loop, the Python source code is executed first in the dynamic graph, and then the statements in the for loop are continuously executed based on the number of loops, and automatic differentiation operations are performed on the operators.

```python
# Set the running mode to dynamic graph mode.
ms.set_context(mode=ms.PYNATIVE_MODE)

class Net(nn.Cell):
    """Customize a network."""
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x):
        for _ in range(3):
            x = x + self.z
        return x

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
net = Net()
output = net(x)

print(output)
```

```text
    [4. 5. 6.]
```
