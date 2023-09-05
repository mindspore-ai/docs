# 使用静态图加速

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/accelerate_with_static_graph.md)

## 背景介绍

AI编译框架分为两种运行模式，分别是动态图模式以及静态图模式。MindSpore默认情况下是以动态图模式运行，但也支持手工切换为静态图模式。两种运行模式的详细介绍如下：

### 动态图模式

动态图的特点是计算图的构建和计算同时发生（Define by run），其符合Python的解释执行方式，在计算图中定义一个Tensor时，其值就已经被计算且确定，因此在调试模型时较为方便，能够实时得到中间结果的值，但由于所有节点都需要被保存，导致难以对整个计算图进行优化。

在MindSpore中，动态图模式又被称为PyNative模式。由于动态图的解释执行特性，在脚本开发和网络流程调试过程中，推荐使用动态图模式进行调试。
如需要手动控制框架采用PyNative模式，可以通过以下代码进行网络构建：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE)  # 使用set_context进行动态图模式的配置

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

### 静态图模式

相较于动态图而言，静态图的特点是将计算图的构建和实际计算分开（Define and run）。有关静态图模式的运行原理，可以参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#概述)。

在MindSpore中，静态图模式又被称为Graph模式，在Graph模式下，基于图优化、计算图整图下沉等技术，编译器可以针对图进行全局的优化，获得较好的性能，因此比较适合网络固定且需要高性能的场景。

如需要手动控制框架采用静态图模式，可以通过以下代码进行网络构建：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.GRAPH_MODE)  # 使用set_context进行运行静态图模式的配置

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

## 静态图模式的使用场景

MindSpore编译器重点面向Tensor数据的计算以及其微分处理。因此使用MindSpore API以及基于Tensor对象的操作更适合使用静态图编译优化。其他操作虽然可以部分入图编译，但实际优化作用有限。另外，静态图模式先编译后执行的模式导致其存在编译耗时。因此，如果函数无需反复执行，那么使用静态图加速也可能没有价值。

有关使用静态图来进行网络编译的示例，请参考[网络构建](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/model.html)。

## 静态图模式开启方式

通常情况下，由于动态图的灵活性，我们会选择使用PyNative模式来进行自由的神经网络构建，以实现模型的创新和优化。但是当需要进行性能加速时，我们需要对神经网络部分或整体进行加速。MindSpore提供了两种切换为图模式的方式，分别是基于装饰器的开启方式以及基于全局context的开启方式。

### 基于装饰器的开启方式

MindSpore提供了jit装饰器，可以通过修饰Python函数或者Python类的成员函数使其被编译成计算图，通过图优化等技术提高运行速度。此时我们可以简单的对想要进行性能优化的模块进行图编译加速，而模型其他部分，仍旧使用解释执行方式，不丢失动态图的灵活性。无论全局context是设置成静态图模式还是动态图模式，被jit修饰的部分始终会以静态图模式进行运行。

在需要对Tensor的某些运算进行编译加速时，可以在其定义的函数上使用jit修饰器，在调用该函数时，该模块自动被编译为静态图。需要注意的是，jit装饰器只能用来修饰函数，无法对类进行修饰。jit的使用示例如下：

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

@ms.jit  # 使用ms.jit装饰器，使被装饰的函数以静态图模式运行
def run(x):
    model = Network()
    return model(x)

output = run(input)
print(output)
```

除使用修饰器外，也可使用函数变换方式调用jit方法，示例如下：

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

run_with_jit = ms.jit(run)  # 通过调用jit将函数转换为以静态图方式执行
output = run(input)
print(output)
```

当我们需要对神经网络的某部分进行加速时，可以直接在construct方法上使用jit修饰器，在调用实例化对象时，该模块自动被编译为静态图。示例如下：

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

    @ms.jit  # 使用ms.jit装饰器，使被装饰的函数以静态图模式运行
    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
model = Network()
output = model(input)
print(output)
```

### 基于context的开启方式

context模式是一种全局的设置模式。代码示例如下：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.GRAPH_MODE)  # 使用set_context进行运行静态图模式的配置

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

## 静态图的语法约束

在Graph模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。因此，编译器无法支持全量的Python语法。MindSpore的静态图编译器维护了Python常用语法子集，以支持神经网络的构建及训练。详情可参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)。

同时，在定义网络时使用流程控制语句时会有部分特殊约束，详情可参考[流程控制语句](https://mindspore.cn/tutorials/experts/zh-CN/master/network/control_flow.html)。

## JitConfig配置选项

在图模式下，可以通过使用[JitConfig](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.JitConfig.html#mindspore.JitConfig)配置选项来一定程度的自定义编译流程，目前JitConfig支持的配置参数如下：

- jit_level: 用于控制优化等级。
- exec_mode: 用于控制模型执行方式。
- jit_syntax_level: 设置静态图语法支持级别，详细介绍请见[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#概述)。

## 静态图高级编程技巧

使用静态图高级编程技巧可以有效地提高编译效率以及执行效率，并可以使程序运行的更加稳定。详情可参考[静态图高级编程技巧](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/optimize/static_graph_expert_programming.md)。