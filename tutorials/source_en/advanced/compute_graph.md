<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/compute_graph.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

# Computational Graph

![comp-graph.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/images/comp-graph.png)

A computational graph is a form of representing mathematical expressions by using directed graphs. As shown in the figure, the neural network structure can be regarded as a computational graph consisting of Tensor data and Tensor operations as nodes, so constructing a neural network and training it by using a deep learning framework is the process of constructing a computational graph and executing it. The current support for computational graphs in the industry framework is divided into two modes: dynamic graphs are executed by interpretation, with dynamic syntax affinity and flexible expression, and static graphs are executed by using JIT (just in time) compilation optimization. There are more restrictions on the syntax for using static syntax.

MindSpore supports both computational graph modes with a unified API expression, using the same API in both modes and a unified automatic differentiation mechanism to achieve the unification of dynamic and static graphs. In the following, we introduce each of the two computational graph modes of MindSpore.

## Dynamic Graphs

Dynamic graphs are characterized by the fact that the construction and computation of a computational graph occur simultaneously (Define by run), which is consistent with Python interpreted execution. When a Tensor is defined in the computational graph, its value is calculated and determined, so it is easier to debug the model and get the value of intermediate results in real time. However, the need for all nodes to be saved makes it difficult to optimize the entire computational graph.

In MindSpore, the dynamic graph pattern is also known as the PyNative pattern. Due to interpreted execution of dynamic graphs, it is recommended to use dynamic graph mode for debugging during script development and network process debugging.

> Default computational graph mode in MindSpore is PyNative mode.

If you need to manually control the framework to adopt PyNative mode, you can configure it with the following code:

```python
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)
```

In PyNative mode, the underlying operaors corresponding to all computation nodes is executed in a single Kernel, so that printing and debugging of computation results can be done arbitrarily, e.g.

```python
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.random.randn(5, 3), ms.float32), name='w') # weight
        self.b = Parameter(Tensor(np.random.randn(3,), ms.float32), name='b') # bias

    def construct(self, x):
        out = ops.matmul(x, self.w)
        print('matmul: ', out)
        out = out + self.b
        print('add bias: ', out)
        return out

model = Network()
```

We simply define a Tensor with a shape of (5,) as input and observe the output. You can see that the `print` statement inserted in the `construct` method prints out the intermediate results in real time.

```python
x = ops.ones(5, ms.float32)  # input tensor
```

```python
out = model(x)
print('out: ', out)
```

```text
matmul:  [-1.8809001   2.0400267   0.32370526]
add bias:  [-1.6770952   1.5087128   0.15726662]
out:  [-1.6770952   1.5087128   0.15726662]
```

## Static Graphs

Compared to dynamic graphs, static graphs separate the construction of the computational graph from the actual computation (Define and run). In the build phase, the original computational graph is optimized and tuned according to the complete computational flow, and compiled to obtain a more memory-efficient and less computationally intensive computational graph. Since the structure of the graph does not change after compilation, it is called a "static graph". In the calculation phase, the results are obtained by executing the compiled calculation graph based on the input data. Compared with dynamic graphs, static graphs have a richer grasp of global information and can be optimized more, but their intermediate processes are black boxes for users, and they cannot get the intermediate calculation results in real time like dynamic graphs.

In MindSpore, the static graph mode is also known as Graph mode. In Graph mode, based on graph optimization, computational graph whole graph sink and other techniques, the compiler can perform global optimization for the graph and obtain better performance, so it is more suitable for scenarios where the network is fixed and high performance is required.

If you need to manually control the framework to adopt Graph mode, you can configure it with the following code:

```python
ms.set_context(mode=ms.GRAPH_MODE)
```

### Graph Compilation Based on Source Code Conversion

In static graph mode, MindSpore converts Python source code into Intermediate Representation IR (IR) by means of source code conversion, and based on this, optimizes the IR graph, and finally executes the optimized graph on hardware devices. MindSpore uses a functional IR based on graph representation, called MindIR. For details, see [Intermediate Representation MindIR](https://www.mindspore.cn/docs/en/master/design/mindir.html).

MindSpore static graph execution process actually consists of two steps, corresponding to the Define and Run phases of the static graph. However, in practice, it is not sensed when the instantiated Cell object is called. MindSpore encapsulates both phases in the `__call__` method of the Cell, so the actual calling process is as follows:

`model(inputs) = model.compile(inputs) + model.construct(inputs)`, where `model` is the instantiated Cell object.

We call the `compile` method explicitly for the following example:

```python
model = Network()

model.compile(x)
out = model(x)
print('out: ', out)
```

```text
out:  [-0.26551223  3.0243678   0.706525  ]
```

### Static Graph Syntax

In Graph mode, Python code is not executed by the Python interpreter, but the code is compiled into a static computational graph, and then the static computational graph is executed. Therefore, the compiler cannot support the full amount of Python syntax. MindSpore static graph compiler maintains a subset of Python common syntax to support the construction and training of neural networks. For details, refer to [Static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Static Graph Control Flow

In PyNative mode, MindSpore fully supports flow control statements in Python native syntax. In Graph mode, MindSpore is compiled with performance optimizations, so there are some special constraints on the use of flow control statements when defining networks, but the rest remains consistent with the native Python syntax. For details, refer to [flow control statements](https://mindspore.cn/tutorials/experts/en/master/network/control_flow.html).

## Just-in-time Compilation

Usually, due to the flexibility of dynamic graphs, we choose to use the PyNative model for free neural network construction to achieve model innovation and optimization. But when performance acceleration is needed, we need to accelerate the neural network in part or as a whole. At this point, switching directly to Graph mode is an easy option, but the limitations of static graphs on syntax and control flow make it impossible to convert from dynamic to static graphs senselessly.

For this purpose, MindSpore provides the `jit` decorator, which can make Python functions or Python-class member functions compiled into computational graphs, and improves the running speed by graph optimization and other techniques. At this point we can simply accelerate the graph compilation for the modules we want to optimize for performance, while the rest of the model, we still use the interpreted execution method, without losing the flexibility of dynamic graphs.

### Cell Module Compilation

When we need to speed up a part of the neural network, we can use the `jit` modifier directly on the `construct` method. The module is automatically compiled to a static graph when the instantiated object is called. The example is as follows:

```python
import mindspore as ms
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(10, 1)

    @ms.jit
    def construct(self, x):
        return self.fc(x)
```

### Function Compilation

Similar to the Cell module compilation, when you need to compile acceleration for certain operations of Tensor, you can use `jit` modifier on its defined function. The module is automatically compiled as a static graph when the function is called. An example is as follows:

> Based on the functional auto-differentiation feature of MindSpore, it is recommended to use the function compilation method for JIT compilation acceleration of Tensor operations.

```python
@ms.jit
def mul(x, y):
    return x * y
```

### Whole-graph Compilation

MindSpore supports compiling and optimizing the forward computation, back propagation, and gradient optimization update of neural network training into one computational graph, which is called whole graph compilation. At this point, it only needs to construct the neural network training logic as a function and use the `jit` modifier on the function to achieve whole-graph compilation.

The following is an example by using a simple fully connected network:

```python
network = nn.Dense(10, 1)
loss_fn = nn.BCELoss()
optimizer = nn.Adam(network.trainable_params(), 0.01)

def forward_fn(data, label):
    logits = network(data)
    loss = loss_fn(logits, label)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

@ms.jit
def train_step(data, label):
    loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

As shown in the above code, after encapsulating the neural network forward execution and loss function as `forward_fn`, the function transformation is executed to obtain the gradient calculation function. Then the gradient calculation function and optimizer calls are encapsulated as `train_step` function and modified with `jit`. When the `train_step` function is called, the static graph is compiled, the whole graph is obtained and executed.

In addition to using modifiers, `jit` methods can also be called by using function transformations, as in the following example:

```python
train_step = ms.jit(train_step)
```
