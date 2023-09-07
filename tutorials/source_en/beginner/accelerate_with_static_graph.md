# Accelerating with Static Graphs

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/accelerate_with_static_graph.md)

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/autograd.md) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || **Accelerating with Static Graphs**

## Background

The AI compilation framework is divided into two modes of operation, namely, dynamic graph mode and static graph mode. MindSpore runs in dynamic graph mode by default, but it also supports manual switching to static graph mode. The details of the two modes are as follows:

### Dynamic Graph Mode

Dynamic graphs are characterized by the construction of the computational graph and computation occurring at the same time (Define by run), which is in line with Python interpreted execution. When defining a Tensor in the computational graph, its value is computed and determined, so it is more convenient to debug the model, and can be able to get the value of the intermediate results in real time, but it is difficult to optimize the whole computational graph due to the fact that all the nodes need to be saved.

In MindSpore, dynamic graph mode is also known as PyNative mode. Due to the interpreted execution of dynamic graphs, it is recommended to use dynamic graph mode for debugging during script development and network process debugging.
If you need to manually control the framework to use PyNative mode, you can configure it with the following code:

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
```

In PyNative mode, the underlying operator corresponding to all computational nodes is executed using a single Kernel, so that printing and debugging of computational results can be done arbitrarily. For example:

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
x = ops.ones(5, ms.float32)
out = model(x)
print("out: ", out)
```

We simply define a Tensor with a shape of (5,) as input and observe the output. You can see that the `print` statement inserted in the `construct` method prints out the intermediate results in real time.

```text
matmul:  [-1.8809001   2.0400267   0.32370526]
add bias:  [-1.6770952   1.5087128   0.15726662]
out:  [-1.6770952   1.5087128   0.15726662]
```

### Static Graph Mode

Compared to dynamic graphs, static graphs are characterized by separating the construction of the computational graph from the actual computation (Define and run). In the construction phase, the original computational graph is optimized and adjusted according to the complete computational flow, and compiled to obtain a more memory-saving and less computationally intensive computational graph. Since the structure of the graph does not change after compilation, it is called "static graph" . In the computation phase, the compiled computation graph is executed according to the input data to get the computation result. Compared to dynamic graphs, static graphs have richer information about the global information and more optimizations can be done, but their intermediate process is a black box for users, and they cannot get the intermediate computation results in real time like dynamic graphs.

In MindSpore, the static graph mode is also known as Graph mode. In Graph mode, based on techniques such as graph optimization and whole computational graph sinking, the compiler can globally optimize for graphs and obtain better performance, so it is more suitable for scenarios where the network is fixed and high performance is required.

In static graph mode, MindSpore converts Python source code into Intermediate Representation (IR) by means of source conversion, and optimizes the IR graph on this basis, and finally executes the optimized graph on the hardware device.MindSpore uses functional IR based on the graph representation that is called MindIR. For more details, see [Intermediate Representation MindIR](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir).

MindSpore static graph execution process actually consists of two steps, corresponding to the Define and Run phases of the static graph, but in practice, it is not perceived when the instantiated Cell object is called, and MindSpore encapsulates both phases in the Cell's `__call__` method, so the actual calling process is:

`model(inputs) = model.compile(inputs) + model.construct(inputs)`, where `model` instantiates the Cell object.

Below we explicitly call the `compile` method for an example:

```python
model = Network()

model.compile(x)
out = model(x)
print('out: ', out)
```

The result is as follows:

```text
matmul:
Tensor(shape=[3], dtype=Float32, value=[-4.01971531e+00 -5.79053342e-01  3.41115999e+00])
add bias:
Tensor(shape=[3], dtype=Float32, value=[-3.94732714e+00 -1.46257186e+00  4.50144434e+00])
out:  [-3.9473271 -1.4625719  4.5014443]
```

## Scenarios for Static Graph Mode

The MindSpore compiler is focused on the computation of Tensor data and its differential processing. Therefore operations using the MindSpore API and based on Tensor objects are more suitable for static graph compilation optimization. Other operations can be partially compiled into the graph, but the actual optimization is limited. In addition, the static graph mode compiles first and then executes, resulting in compilation time consumption. As a result, there may be no need to use static graph acceleration if the function does not need to be executed repeatedly.

For an example of using static graphs for network compilation, see [Network Build](https://www.mindspore.cn/tutorials/en/master/beginner/model.html).

## Static Graph Mode Startup Method

Usually, due to the flexibility of dynamic graphs, we choose to use PyNative mode for free neural network construction for model innovation and optimization. But when performance acceleration is needed, we need to accelerate the neural network partially or as a whole. MindSpore provides two ways of switching to graph mode, the decorator-based startup method and the global context-based startup method.

### Decorator-based Startup Method

MindSpore provides a jit decorator that can be used to modify Python functions or member functions of Python classes so that they can be compiled into computational graphs, which improves the speed of operation through graph optimization and other techniques. At this point we can simply accelerate the graph compilation for the modules we want to optimize for performance, while the rest of the model, which still uses interpreted execution, does not lose the flexibility of dynamic graphs.

When you need to accelerate the compilation of some of Tensor operations, you can use the jit decorator on the function it defines, and the module is automatically compiled into a static graph when the function is called. The example is as follows:

```python
@ms.jit
def mul(x, y):
    return x * y
```

When we need to accelerate a part of the neural network, we can use the jit decorator directly on the construct method, and the module is automatically compiled as a static graph when the instantiated object is called. The example is as follows:

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

MindSpore supports combining the forward computation, backward propagation, and gradient optimization updating of neural network training into one computational graph for compilation and optimization, which is called whole graph compilation. In this case, you only need to construct the neural network training logic as a function, and use the jit decorator on the function to achieve the effect of whole graph compilation. The following is an example using a simple fully connected network:

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

As shown in the above code, after encapsulating the neural network forward execution and loss function into forward_fn, the function transformation is performed to obtain the gradient calculation function. Then encapsulate the gradient calculation function and optimizer call into train_step function, and use jit to modify it. When calling train_step function, it will perform static graph compilation, get the whole graph and execute it.

In addition to using decorator, jit methods can also be called using function transformations, as shown in the following example:

```python
train_step = ms.jit(train_step)
```

### Context-based Startup Method

The context mode is a global setting mode. The code example is as follows:

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)
```

## Syntax Constraints for Static Graph

In Graph mode, Python code is not executed by the Python interpreter, but the code is compiled into a static computational graph and then the static computational graph is executed. As a result, the compiler cannot support the global Python syntax. MindSpore static graph compiler maintains a subset of common Python syntax to support neural network construction and training. For more details, see [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

## JitConfig Configuration Option

In graph mode, the compilation process can be customized by using the [JitConfig](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.JitConfig.html#mindspore.JitConfig) configuration option. Currently JitConfig supports the following configuration parameters:

- jit_level: Used to control the optimization level.
- exec_mode: Used to control the model execution.
- jit_syntax_level: Set the static graph syntax support level. See [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#overview) for details.

## Advanced Programming Techniques for Static Graphs

Using static graph advanced programming techniques can effectively improve the compilation efficiency as well as the execution efficiency, and can make the program run more stably. For details, please refer to [Advanced Programming Techniques with Static Graphs](https://www.mindspore.cn/tutorials/en/master/advanced/static_graph_expert_programming.html).
