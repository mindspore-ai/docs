[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/accelerate_with_static_graph.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/autograd.md) || [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html) || **Accelerating with Static Graphs**|| [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

# Accelerating with Static Graphs

## Background

The AI compilation framework is divided into two modes of operation: dynamic graph mode and static graph mode. MindSpore runs in dynamic graph mode by default, but it also supports manual switching to static graph mode. The details of the two modes are as follows:

### Dynamic Graph Mode

Dynamic graphs are characterized by the construction of the computational graph and computation occurring at the same time (Define by run), which is in line with Python interpreted execution. When defining a Tensor in the computational graph, its value is computed and determined, so it is more convenient to debug the model, and can be able to get the value of the intermediate results in real time, but it is difficult to optimize the whole computational graph due to the fact that all the nodes need to be saved.

In MindSpore, dynamic graph mode is also known as PyNative mode. Due to the interpreted execution of dynamic graphs, it is recommended to use dynamic graph mode for debugging during script development and network process debugging.
If you need to manually control the framework to use PyNative mode, you can configure it with the following code:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE)  # Dynamic graph mode configuration using set_context

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
output = model(input)
print(output)
```

```text
[[-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 ...
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]
 [-0.00134926 -0.13563682 -0.02863023 -0.05452826  0.03290743 -0.12423715
  -0.0582641  -0.10854103 -0.08558805  0.06099342]]
```

### Static Graph Mode

Compared to dynamic graphs, static graphs are characterized by separating the construction of the computational graph from the actual computation (Define and run). For more information on how the static graph model works, see [Static Graph Syntax Support](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html#overview).

In MindSpore, the static graph mode is also known as Graph mode. In Graph mode, based on techniques such as graph optimization and whole computational graph sinking, the compiler can globally optimize for graphs and obtain better performance, so it is more suitable for scenarios where the network is fixed and high performance is required.

If you need to manually control the framework to use static graph mode, you can build the network with the following code:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.GRAPH_MODE)  # Static graph mode configuration using set_context

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
output = model(input)
print(output)
```

```text
[[ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 ...
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]
 [ 0.05363735  0.05117104 -0.03343301  0.06347139  0.07546629  0.03263091
   0.02790363  0.06269836  0.01838502  0.04387159]]
```

## Scenarios for Static Graph Mode

The MindSpore compiler is focused on the computation of Tensor data and its differential processing. Therefore operations using the MindSpore API and based on Tensor objects are more suitable for static graph compilation optimization. Other operations can be partially compiled into the graph, but the actual optimization is limited. In addition, the static graph mode compiles first and then executes, resulting in compilation time consumption. As a result, there may be no need to use static graph acceleration if the function does not need to be executed repeatedly.

For an example of using static graphs for network compilation, see [Network Build](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html).

## Static Graph Mode Startup Method

Usually, due to the flexibility of dynamic graphs, we choose to use PyNative mode for free neural network construction for model innovation and optimization. But when performance acceleration is needed, we need to accelerate the neural network partially or as a whole. MindSpore provides two ways of switching to graph mode, the decorator-based startup method and the global context-based startup method.

### Decorator-based Startup Method

MindSpore provides a jit decorator that can be used to modify Python functions or member functions of Python classes so that they can be compiled into computational graphs, which improves the speed of operation through graph optimization and other techniques. At this point we can simply accelerate the graph compilation for the modules we want to optimize for performance, while the rest of the model, which still uses interpreted execution, does not lose the flexibility of dynamic graphs. Regardless of whether the global context is set to static graph mode or dynamic graph mode, the part modified by the jit will always run in static graph mode.

When you need to accelerate the compilation of some of Tensor operations, you can use the jit decorator on the function it defines, and the module is automatically compiled into a static graph when the function is called. Note that jit decorators can only be used to modify functions, not classes. The example is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))

@ms.jit  # Use the ms.jit decorator to make the decorated function run in static graph mode
def run(x):
    model = Network()
    return model(x)

output = run(input)
print(output)
```

```text
[[-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 ...
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]
 [-0.12126954  0.06986676 -0.2230821  -0.07087803 -0.01003947  0.01063392
   0.10143848 -0.0200909  -0.09724037  0.0114444 ]]
```

In addition to using modifiers, jit methods can also be called using function transformations, as shown in the following example:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))

def run(x):
    model = Network()
    return model(x)

run_with_jit = ms.jit(run)  # Transforming a function to execute as a static graph by calling jit
output = run_with_jit(input)
print(output)
```

```text
[[ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 ...
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]
 [ 0.11027216 -0.09628229  0.0457969   0.05396656 -0.06958974  0.0428197
  -0.1572069  -0.14151613 -0.04531277  0.07521383]]
```

When we need to accelerate a part of the neural network, we can use the jit modifier directly on the construct method, and the module is automatically compiled as a static graph when the instantiated object is called. The example is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    @ms.jit  # Use the ms.jit decorator to make the decorated function run in static graph mode
    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
model = Network()
output = model(input)
print(output)
```

```text
[[ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 ...
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]
 [ 0.10522258  0.06597593 -0.09440921 -0.04883489  0.07194916  0.1343117
  -0.06813788  0.01986085  0.0216996  -0.05345828]]
```

### Context-based Startup Method

The context mode is a global setting mode. The code example is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.GRAPH_MODE)  # Configuration for running static graph mode using set_context

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
output = model(input)
print(output)
```

```text
[[ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 ...
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]
 [ 0.08501796 -0.04404321 -0.05165704  0.00357929  0.00051521  0.00946456
   0.02748473 -0.19415936 -0.00278988  0.04024826]]
```

## Syntax Constraints for Static Graph

In Graph mode, Python code is not executed by the Python interpreter, but the code is compiled into a static computational graph and then the static computational graph is executed. As a result, the compiler cannot support the global Python syntax. MindSpore static graph compiler maintains a subset of common Python syntax to support neural network construction and training. For more details, see [Static Graph Syntax Support](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html).

## Advanced Programming Techniques for Static Graphs

Using static graph advanced programming techniques can effectively improve the compilation efficiency as well as the execution efficiency, and can make the program run more stably. For details, please refer to [Advanced Programming Techniques with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph_expert_programming.html).
