# 使用静态图加速

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/accelerate_with_static_graph.md)

## 背景介绍

AI编译框架分为两种运行模式，分别是动态图模式以及静态图模式。MindSpore默认情况下是以动态图模式运行，但也支持手工切换为静态图模式。两种运行模式的详细介绍如下：

### 动态图模式

动态图的特点是计算图的构建和计算同时发生（Define by run），其符合Python的解释执行方式，在计算图中定义一个Tensor时，其值就已经被计算且确定，因此在调试模型时较为方便，能够实时得到中间结果的值，但由于所有节点都需要被保存，导致难以对整个计算图进行优化。

在MindSpore中，动态图模式又被称为PyNative模式。由于动态图的解释执行特性，在脚本开发和网络流程调试过程中，推荐使用动态图模式进行调试。
如需要手动控制框架采用PyNative模式，可以通过以下代码进行配置：

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
```

在PyNative模式下，所有计算节点对应的底层算子均采用单Kernel执行的方式，因此可以任意进行计算结果的打印和调试，如：

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

我们简单定义一个shape为(5,)的Tensor作为输入，观察输出情况。可以看到在`construct`方法中插入的`print`语句将中间结果进行实时的打印输出。

```text
matmul:  [-1.8809001   2.0400267   0.32370526]
add bias:  [-1.6770952   1.5087128   0.15726662]
out:  [-1.6770952   1.5087128   0.15726662]
```

### 静态图模式

相较于动态图而言，静态图的特点是将计算图的构建和实际计算分开（Define and run）。在构建阶段，根据完整的计算流程对原始的计算图进行优化和调整，编译得到更省内存和计算量更少的计算图。由于编译之后图的结构不再改变，所以称之为 “静态图” 。在计算阶段，根据输入数据执行编译好的计算图得到计算结果。相较于动态图，静态图对全局的信息掌握更丰富，可做的优化也会更多，但是其中间过程对于用户来说是黑盒，无法像动态图一样实时拿到中间计算结果。

在MindSpore中，静态图模式又被称为Graph模式，在Graph模式下，基于图优化、计算图整图下沉等技术，编译器可以针对图进行全局的优化，获得较好的性能，因此比较适合网络固定且需要高性能的场景。

在静态图模式下，MindSpore通过源码转换的方式，将Python的源码转换成中间表达IR（Intermediate Representation），并在此基础上对IR图进行优化，最终在硬件设备上执行优化后的图。MindSpore使用基于图表示的函数式IR，称为MindIR，详情可参考[中间表示MindIR](https://www.mindspore.cn/docs/zh-CN/master/design/all_scenarios.html#中间表示mindir)。

MindSpore的静态图执行过程实际包含两步，对应静态图的Define和Run阶段，但在实际使用中，在实例化的Cell对象被调用时并不会感知，MindSpore将两阶段均封装在Cell的`__call__`方法中，因此实际调用过程为：

`model(inputs) = model.compile(inputs) + model.construct(inputs)`，其中`model`为实例化Cell对象。

下面我们显式调用`compile`方法进行示例：

```python
model = Network()

model.compile(x)
out = model(x)
print('out: ', out)
```

结果如下：

```text
matmul:
Tensor(shape=[3], dtype=Float32, value=[-4.01971531e+00 -5.79053342e-01  3.41115999e+00])
add bias:
Tensor(shape=[3], dtype=Float32, value=[-3.94732714e+00 -1.46257186e+00  4.50144434e+00])
out:  [-3.9473271 -1.4625719  4.5014443]
```

## 静态图模式的使用场景

MindSpore编译器重点面向Tensor数据的计算以及其微分处理。因此使用MindSpore API以及基于Tensor对象的操作更适合使用静态图编译优化。其他操作虽然可以部分入图编译，但实际优化作用有限。另外，静态图模式先编译后执行的模式导致其存在编译耗时。因此，如果函数无需反复执行，那么使用静态图加速也可能没有价值。

有关使用静态图来进行网络编译的示例，请参考[网络构建](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/model.html)。

## 静态图模式开启方式

通常情况下，由于动态图的灵活性，我们会选择使用PyNative模式来进行自由的神经网络构建，以实现模型的创新和优化。但是当需要进行性能加速时，我们需要对神经网络部分或整体进行加速。MindSpore提供了两种切换为图模式的方式，分别是基于装饰器的开启方式以及基于全局context的开启方式。

### 基于装饰器的开启方式

MindSpore提供了jit装饰器，可以通过修饰Python函数或者Python类的成员函数使其被编译成计算图，通过图优化等技术提高运行速度。此时我们可以简单的对想要进行性能优化的模块进行图编译加速，而模型其他部分，仍旧使用解释执行方式，不丢失动态图的灵活性。

在需要对Tensor的某些运算进行编译加速时，可以在其定义的函数上使用jit修饰器，在调用该函数时，该模块自动被编译为静态图。示例如下：

```python
@ms.jit
def mul(x, y):
    return x * y
```

当我们需要对神经网络的某部分进行加速时，可以直接在construct方法上使用jit修饰器，在调用实例化对象时，该模块自动被编译为静态图。示例如下：

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

MindSpore支持将神经网络训练的正向计算、反向传播、梯度优化更新等步骤合为一个计算图进行编译优化，此方法称为整图编译。此时，仅需将神经网络训练逻辑构造为函数，并在函数上使用jit修饰器，即可达到整图编译的效果。下面使用简单的全连接网络进行举例：

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

如上述代码所示，将神经网络正向执行与损失函数封装为forward_fn后，执行函数变换获得梯度计算函数。而后将梯度计算函数、优化器调用封装为train_step函数，并使用jit进行修饰，调用train_step函数时，会进行静态图编译，获得整图并执行。

除使用修饰器外，也可使用函数变换方式调用jit方法，示例如下：

```python
train_step = ms.jit(train_step)
```

### 基于context的开启方式

context模式是一种全局的设置模式。代码示例如下：

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)
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